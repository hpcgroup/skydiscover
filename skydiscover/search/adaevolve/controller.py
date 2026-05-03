"""
AdaEvolve Controller - Evolution loop with adaptive search intensity.

A clean implementation that uses the adaptive database for all
exploration/exploitation decisions. No explicit stagnation tracking -
search intensity handles exploration automatically.

Features:
- Adaptive sampling based on accumulated improvement signal
- Mode-aware prompting (exploration vs exploitation)
- Paradigm breakthrough for high-level strategy shifts
- Sibling context for learning from previous attempts
- Comprehensive JSON logging of all AdaEvolve signals
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from skydiscover.context_builder.adaevolve import AdaEvolveContextBuilder
from skydiscover.context_builder.default import DefaultContextBuilder
from skydiscover.evaluation.llm_judge import LLMJudge
from skydiscover.llm.llm_pool import LLMPool
from skydiscover.search.adaevolve.paradigm import ParadigmGenerator
from skydiscover.search.base_database import Program
from skydiscover.search.default_discovery_controller import (
    DiscoveryController,
    DiscoveryControllerInput,
)
from skydiscover.search.utils.discovery_utils import SerializableResult
from skydiscover.utils.code_utils import (
    apply_diff,
    extract_diffs,
    format_diff_summary,
    parse_full_rewrite,
)

logger = logging.getLogger(__name__)


class AdaEvolveController(DiscoveryController):
    """
    AdaEvolve evolution controller with adaptive search intensity.

    Key Features:
    1. Adaptive sampling: Search intensity per island determines exploration/exploitation
    2. Mode-aware prompting: Different guidance for exploration vs exploitation
    3. Sibling context: Shows previous mutations for learning
    4. Error retry: Retries failed generations with error context
    5. Island rotation: UCB-based selection via database.end_iteration()
    6. Paradigm breakthrough: High-level strategy shifts when globally stuck

    No explicit stagnation tracking - search intensity handles exploration
    automatically based on accumulated improvement signal.
    """

    def __init__(self, controller_input: DiscoveryControllerInput):
        super().__init__(controller_input)

        # Configuration
        db_config = self.config.search.database
        self.enable_retry = getattr(db_config, "enable_error_retry", True)
        self.max_retries = getattr(db_config, "max_error_retries", 2)
        self.num_context_programs = self.config.search.num_context_programs

        # Components
        self.llms = LLMPool(self.config.llm.models)
        self.context_builder = AdaEvolveContextBuilder(self.config)

        # Paradigm generator (if paradigm breakthrough is enabled)
        # Note: We check database.use_paradigm_breakthrough at runtime, not this init-time flag
        # This ensures correct behavior after checkpoint load
        if self.database.use_paradigm_breakthrough:
            model_names = ", ".join(m.name for m in self.guide_llms.models_cfg)
            logger.info(f"Paradigm LLM: using guide_models [{model_names}]")

            self.paradigm_generator = ParadigmGenerator(
                llm_pool=self.guide_llms,
                system_message=self.config.context_builder.system_message or "",
                evaluator_code=self._load_evaluator_code(),
                num_paradigms=self.database.get_paradigm_num_to_generate(),
                eval_timeout=self.config.evaluator.timeout,
                language=self.config.language or "python",
                objective_names=getattr(db_config, "pareto_objectives", []),
                higher_is_better=getattr(db_config, "higher_is_better", {}),
                fitness_key=getattr(db_config, "fitness_key", None),
            )
        else:
            self.paradigm_generator = None

        # JSON logging for comprehensive AdaEvolve stats
        self._iteration_stats_log_path: Optional[str] = None
        self._iteration_stats_file = None
        self._last_sampling_mode: Optional[str] = None
        self._last_sampling_intensity: Optional[float] = None

        # Parallel-iteration coordination. Lazily-initialised because asyncio
        # locks must be created on the running event loop.
        self._scheduler_lock: Optional[asyncio.Lock] = None
        self._paradigm_lock: Optional[asyncio.Lock] = None
        # Tracks the highest contiguous iteration that has fully completed,
        # used by the parallel loop to fire checkpoints in iteration order
        # despite out-of-order task completion.
        self._completed_iterations: set = set()
        self._highest_contiguous_completed: int = -1
        self._last_checkpoint_iteration: int = -1

        logger.info(
            f"AdaEvolveController initialized "
            f"(language={self.config.language}, "
            f"paradigm_breakthrough={self.database.use_paradigm_breakthrough})"
        )

    def _load_evaluator_code(self) -> str:
        """Load evaluator source code for paradigm generation context."""
        from skydiscover.search.utils.discovery_utils import load_evaluator_code

        return load_evaluator_code(self.evaluation_file)

    # =========================================================================
    # JSON Logging for AdaEvolve Stats
    # =========================================================================

    def _setup_iteration_stats_logging(self, output_dir: Optional[str] = None) -> None:
        """
        Set up JSON logging for comprehensive iteration statistics.

        Creates a JSONL file that records all AdaEvolve signals at each iteration.
        This enables detailed post-hoc analysis of the discovery process.

        Args:
            output_dir: Directory to write the log file. If None, uses database.config.db_path
        """
        # Determine output directory
        if output_dir is None:
            output_dir = self.output_dir
        if output_dir is None:
            output_dir = getattr(self.database.config, "db_path", None)
        if output_dir is None:
            output_dir = "."

        os.makedirs(output_dir, exist_ok=True)

        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._iteration_stats_log_path = os.path.join(
            output_dir, f"adaevolve_iteration_stats_{timestamp}.jsonl"
        )

        logger.info(
            f"AdaEvolve iteration stats will be logged to: {self._iteration_stats_log_path}"
        )

    def _log_iteration_stats(
        self,
        iteration: int,
        sampling_mode: Optional[str] = None,
        sampling_intensity: Optional[float] = None,
        child_program: Optional[Dict] = None,
        iteration_time: Optional[float] = None,
        llm_generation_time: Optional[float] = None,
        eval_time: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Log comprehensive iteration statistics to JSON file.

        This method collects all AdaEvolve signals and writes them as a single
        JSON line to the log file for easy post-processing.

        Args:
            iteration: Current iteration number
            sampling_mode: The mode used for sampling (exploration/exploitation/balanced)
            sampling_intensity: The search intensity value used
            child_program: The child program dict if successfully generated
            iteration_time: Time taken for this iteration
            error: Error message if iteration failed
        """
        if self._iteration_stats_log_path is None:
            return

        try:
            # Get comprehensive stats from database
            stats = self.database.get_comprehensive_iteration_stats(
                iteration=iteration,
                sampling_mode=(
                    sampling_mode if sampling_mode is not None else self._last_sampling_mode
                ),
                sampling_intensity=(
                    sampling_intensity
                    if sampling_intensity is not None
                    else self._last_sampling_intensity
                ),
            )

            # Add timestamp
            stats["timestamp"] = datetime.now().isoformat()

            # Add iteration-specific info
            stats["iteration_result"] = {
                "success": error is None,
                "error": error,
                "iteration_time_seconds": iteration_time,
                "llm_generation_time_seconds": llm_generation_time,
                "eval_time_seconds": eval_time,
            }

            # Add child program info if available
            if child_program:
                stats["iteration_result"]["child_program"] = {
                    "id": child_program.get("id"),
                    "metrics": child_program.get("metrics"),
                    "generation": child_program.get("generation"),
                    "parent_id": child_program.get("parent_id"),
                }

            # Write to JSONL file
            with open(self._iteration_stats_log_path, "a") as f:
                f.write(json.dumps(stats, default=str) + "\n")

        except Exception as e:
            logger.warning(f"Failed to log iteration stats: {e}")

    def get_iteration_stats_log_path(self) -> Optional[str]:
        """Get the path to the iteration stats log file."""
        return self._iteration_stats_log_path

    # =========================================================================
    # Main Evolution Loop
    # =========================================================================

    async def run_discovery(
        self,
        start_iteration: int,
        max_iterations: int,
        checkpoint_callback=None,
    ) -> Optional[Program]:
        """Run evolution with adaptive search intensity and island rotation.

        Sequential by default. When ``config.max_parallel_iterations > 1``
        delegates to ``_run_discovery_parallel`` which runs up to N
        iterations concurrently with distinct-island reservations.
        """
        total = start_iteration + max_iterations
        max_parallel = max(1, getattr(self.config, "max_parallel_iterations", 1))

        logger.info(
            f"AdaEvolve: Running {max_iterations} iterations "
            f"across {self.database.num_islands} islands "
            f"(max_parallel={max_parallel})"
        )

        # Set up comprehensive JSON logging for iteration stats
        self._setup_iteration_stats_logging()

        # Ensure all islands are seeded
        self._ensure_all_islands_seeded()

        if max_parallel > 1:
            await self._run_discovery_parallel(
                start_iteration, total, max_parallel, checkpoint_callback,
            )
        else:
            await self._run_discovery_sequential(
                start_iteration, total, checkpoint_callback,
            )

        logger.info("AdaEvolve completed")
        self.database.log_status()

        # Log final summary and stats file location
        if self._iteration_stats_log_path:
            logger.info(f"AdaEvolve iteration stats saved to: {self._iteration_stats_log_path}")

        return self.database.get_best_program()

    async def _run_discovery_sequential(
        self,
        start_iteration: int,
        total: int,
        checkpoint_callback,
    ) -> None:
        """Original sequential discovery loop (max_parallel_iterations=1)."""
        for iteration in range(start_iteration, total):
            if self.shutdown_event.is_set():
                logger.info("Shutdown requested")
                break

            try:
                await self._run_iteration(iteration, checkpoint_callback)
            except Exception as e:
                logger.exception(f"Iteration {iteration} failed: {e}")
            finally:
                # CRITICAL: Tell database iteration is complete
                # This handles island rotation (UCB) and migration
                self.database.end_iteration(iteration)

    async def _run_discovery_parallel(
        self,
        start_iteration: int,
        total: int,
        max_parallel: int,
        checkpoint_callback,
    ) -> None:
        """Parallel discovery loop with distinct-island reservations.

        Differences from sequential:
        - Up to ``max_parallel`` iterations in flight at once.
        - Each iteration reserves a distinct island via
          ``database.reserve_island()`` under ``_scheduler_lock`` so
          concurrent samples don't cluster on the same island.
        - ``end_iteration()`` and ``release_island()`` run together under
          the same lock so spawn/migrate are atomic w.r.t. concurrent
          reserves.
        - Checkpoints fire from this loop based on the highest contiguous
          completed iteration (iterations finish out of order, but
          checkpoints stay deterministic).
        """
        # Initialize asyncio locks now that we're on the event loop.
        if self._scheduler_lock is None:
            self._scheduler_lock = asyncio.Lock()
        if self._paradigm_lock is None:
            self._paradigm_lock = asyncio.Lock()

        # Reset checkpoint tracking for this run
        self._completed_iterations = set()
        self._highest_contiguous_completed = start_iteration - 1
        self._last_checkpoint_iteration = start_iteration - 1

        sem = asyncio.Semaphore(max_parallel)
        pending: set = set()

        logger.info(
            f"AdaEvolve parallel: up to {max_parallel} iterations in flight "
            f"({start_iteration}..{total - 1})"
        )

        async def _bounded_iteration(iteration: int) -> None:
            """Run one iteration: reserve island, run, release+end_iteration."""
            async with sem:
                if self.shutdown_event.is_set():
                    return

                # Reserve a distinct island (sync, but lock-protected for
                # composition with end_iteration and future-proofing).
                async with self._scheduler_lock:
                    try:
                        island = self.database.reserve_island()
                    except RuntimeError as e:
                        logger.error(f"Iteration {iteration}: {e}")
                        return

                try:
                    # checkpoint_callback=None — parallel loop fires
                    # checkpoints based on highest-contiguous-completed,
                    # not per-iteration.
                    await self._run_iteration(
                        iteration, checkpoint_callback=None, island_idx=island,
                    )
                except Exception as e:
                    logger.exception(f"Iteration {iteration} failed: {e}")
                finally:
                    async with self._scheduler_lock:
                        self.database.release_island(island)
                        # CRITICAL: rotation/spawn/migration. Sync, runs
                        # under the same lock so concurrent reserves can't
                        # observe a half-spawned island.
                        try:
                            self.database.end_iteration(iteration)
                        except Exception as e:
                            logger.exception(
                                f"end_iteration({iteration}) failed: {e}"
                            )

                    # Advance the contiguous-completed pointer and fire any
                    # pending checkpoints in iteration order.
                    self._mark_iteration_complete(iteration, checkpoint_callback)

        # Backpressure: keep the pipeline full but bounded
        for iteration in range(start_iteration, total):
            if self.shutdown_event.is_set():
                break

            task = asyncio.create_task(
                _bounded_iteration(iteration), name=f"iter_{iteration}",
            )
            pending.add(task)
            task.add_done_callback(pending.discard)

            if len(pending) >= max_parallel:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED,
                )
                for t in done:
                    if t.exception() is not None:
                        logger.warning(
                            f"Parallel iteration task failed: {t.exception()}"
                        )

        # Drain remaining tasks
        if pending:
            done, _ = await asyncio.wait(pending)
            for t in done:
                if t.exception() is not None:
                    logger.warning(
                        f"Parallel drain: task failed: {t.exception()}"
                    )

    def _mark_iteration_complete(self, iteration: int, checkpoint_callback) -> None:
        """Track completion and fire checkpoints in iteration order.

        Called from each parallel task after end_iteration. Iterations
        complete out of order, but we want checkpoints to fire
        deterministically. So we maintain a contiguous high-water mark and
        only fire callbacks when the mark advances past a checkpoint
        interval boundary.

        Sync method — atomic between awaits in asyncio.
        """
        self._completed_iterations.add(iteration)

        # Advance the contiguous pointer as far as completed_iterations
        # allows.
        next_iter = self._highest_contiguous_completed + 1
        while next_iter in self._completed_iterations:
            self._completed_iterations.discard(next_iter)
            self._highest_contiguous_completed = next_iter
            next_iter += 1

        if checkpoint_callback is None:
            return

        interval = self.config.checkpoint_interval
        high = self._highest_contiguous_completed
        # Find the largest interval-aligned iteration we've now passed
        # but haven't checkpointed yet.
        target = (high // interval) * interval
        if target > 0 and target > self._last_checkpoint_iteration:
            self._last_checkpoint_iteration = target
            logger.info(
                f"Checkpoint interval reached at iteration {target} "
                f"(contiguous up to {high})"
            )
            self.database.log_status()
            try:
                checkpoint_callback(target)
            except Exception as e:
                logger.exception(f"Checkpoint callback failed: {e}")

    def _ensure_all_islands_seeded(self) -> None:
        """Ensure all islands have at least one program.

        Empty islands receive clones from the available seed pool, distributed
        round-robin so that seed diversity is preserved when num_islands >
        num_seeds. With 2 seeds across 48 islands this yields ~24 clones per
        seed instead of 47 from one and 1 from the other.
        """
        # Collect one program from each already-populated island as the pool.
        seed_pool: List[Program] = []
        for i in range(self.database.num_islands):
            if self.database.get_island_size(i) > 0:
                population = self.database.get_island_population(i)
                if population:
                    seed_pool.append(population[0])

        if not seed_pool:
            logger.warning("No seed program found")
            return

        rr_idx = 0
        for i in range(self.database.num_islands):
            if self.database.get_island_size(i) == 0:
                seed = seed_pool[rr_idx % len(seed_pool)]
                rr_idx += 1
                copy = Program(
                    id=str(uuid.uuid4()),
                    solution=seed.solution,
                    language=seed.language,
                    metrics=seed.metrics.copy() if seed.metrics else {},
                    iteration_found=seed.iteration_found,
                    parent_id=None,
                    generation=0,
                    metadata={"seeded_to_island": i, "cloned_from": seed.id},
                )
                self.database.add(copy, iteration=0, target_island=i)
                logger.info(f"Seeded island {i} (clone of {seed.id[:8]})")

    async def _run_iteration(
        self,
        iteration: int,
        checkpoint_callback,
        island_idx: Optional[int] = None,
    ) -> SerializableResult:
        """Execute one evolution iteration.

        Args:
            island_idx: Explicit island (parallel mode). None = sequential.

        Returns:
            The SerializableResult so the parallel caller can inspect it.
            (Sequential callers can ignore the return value.)
        """
        iteration_start_time = time.time()

        # Check for global paradigm stagnation
        # Use database flag directly to stay in sync after checkpoint load
        if self.database.use_paradigm_breakthrough and self.database.is_paradigm_stagnating():
            await self._generate_paradigms_if_needed()

        result = await self._run_normal_step(iteration, island_idx=island_idx)

        iteration_time = time.time() - iteration_start_time

        # Prefer per-result sampling state (set by _generate_child) over
        # shared self.* fields so parallel iterations log accurate values.
        sampling_mode = result.sampling_mode if result.sampling_mode is not None else self._last_sampling_mode
        sampling_intensity = (
            result.sampling_intensity if result.sampling_intensity is not None
            else self._last_sampling_intensity
        )

        if result.error:
            logger.warning(f"Iteration {iteration}: {result.error}")
            # Log failed iteration stats
            self._log_iteration_stats(
                iteration=iteration,
                sampling_mode=sampling_mode,
                sampling_intensity=sampling_intensity,
                child_program=None,
                iteration_time=iteration_time,
                llm_generation_time=result.llm_generation_time,
                eval_time=result.eval_time,
                error=result.error,
            )
        else:
            self._process_result(result, iteration, checkpoint_callback)
            # Log successful iteration stats
            self._log_iteration_stats(
                iteration=iteration,
                sampling_mode=sampling_mode,
                sampling_intensity=sampling_intensity,
                child_program=result.child_program_dict,
                iteration_time=result.iteration_time,
                llm_generation_time=result.llm_generation_time,
                eval_time=result.eval_time,
                error=None,
            )
        return result

    async def _generate_paradigms_if_needed(self) -> None:
        """Generate new paradigms if stagnating and none active.

        In parallel mode, multiple concurrent iterations can pass the
        ``has_active_paradigm()`` check before any of them sets a paradigm,
        which would spawn N redundant paradigm-LLM calls. ``_paradigm_lock``
        serialises the check + generate sequence across the await.
        """
        if self.paradigm_generator is None:
            return

        if self._paradigm_lock is not None:
            async with self._paradigm_lock:
                if self.database.has_active_paradigm():
                    return  # Another iteration already generated paradigms
                await self._do_generate_paradigms()
            return

        if self.database.has_active_paradigm():
            return  # Already have paradigms to use

        await self._do_generate_paradigms()

    async def _do_generate_paradigms(self) -> None:
        """Inner implementation of paradigm generation (no locking)."""
        logger.info("Global paradigm stagnation detected, generating breakthrough ideas...")

        # Get current best program for context
        best_program = self.database.get_best_program()
        best_solution = best_program.solution if best_program else ""
        best_score = self.database.get_program_proxy_score(best_program)

        # Extract evaluator feedback from the best program's artifacts
        evaluator_feedback = None
        if best_program and best_program.artifacts:
            fb = best_program.artifacts.get("feedback")
            if fb and isinstance(fb, str):
                evaluator_feedback = fb

        # Get previously tried ideas for feedback
        previously_tried = self.database.get_previously_tried_ideas()

        # Generate new paradigms
        paradigms = await self.paradigm_generator.generate(
            current_program_solution=best_solution,
            current_best_score=best_score,
            previously_tried_ideas=previously_tried,
            evaluator_feedback=evaluator_feedback,
        )

        if paradigms:
            self.database.set_paradigms(paradigms)
            logger.info(f"Generated {len(paradigms)} breakthrough paradigms")
        else:
            logger.warning("Failed to generate paradigms")

    async def _run_normal_step(
        self,
        iteration: int,
        island_idx: Optional[int] = None,
    ) -> SerializableResult:
        """Run a normal iteration with optional retry.

        Args:
            island_idx: Explicit island for parallel mode (forwarded to
                ``_generate_child``). None preserves sequential behaviour.
        """
        last_error = None
        attempts = 1 + (self.max_retries if self.enable_retry else 0)

        for attempt in range(attempts):
            result = await self._generate_child(
                iteration, error_context=last_error, island_idx=island_idx,
            )
            if not result.error:
                return result
            last_error = result.error
            logger.debug(f"Attempt {attempt + 1}/{attempts} failed: {last_error}")

        return SerializableResult(
            error=f"All {attempts} attempts failed: {last_error}",
            iteration=iteration,
        )

    def _process_result(
        self,
        result: SerializableResult,
        iteration: int,
        checkpoint_callback,
    ) -> None:
        """Process a successful result by adding to database."""
        child = Program(**result.child_program_dict)

        # Add to database (database handles which island)
        self.database.add(child, iteration=iteration, parent_id=result.parent_id)

        # Fire monitor callback (live dashboard)
        if self.monitor_callback:
            try:
                self.monitor_callback(child, iteration)
            except Exception:
                logger.debug("Monitor callback error", exc_info=True)

        # Log prompt
        if result.prompt:
            self.database.log_prompt(
                template_key=(
                    "full_rewrite_user_message"
                    if not self.config.diff_based_generation
                    else "diff_user_message"
                ),
                program_id=child.id,
                prompt=result.prompt,
                responses=[result.llm_response] if result.llm_response else [],
            )

        # Log progress
        logger.info(
            f"Iteration {iteration}: Program {child.id[:8]} "
            f"(parent: {result.parent_id[:8] if result.parent_id else 'None'}) "
            f"completed in {result.iteration_time:.2f}s"
            f" (llm: {result.llm_generation_time:.2f}s,"
            f" eval: {result.eval_time:.2f}s)"
        )

        # Log metrics
        if child.metrics:
            metrics_str = ", ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in child.metrics.items()
            )
            logger.info(f"Metrics: {metrics_str}")

        # Check for new best
        if self.database.is_multiobjective_enabled():
            pareto_front_ids = {program.id for program in self.database.get_pareto_front()}
            if child.id in pareto_front_ids:
                logger.info(f"Program entered the global Pareto front at iteration {iteration}")
            if self.database.best_program_id == child.id:
                logger.info(f"New representative Pareto solution found at iteration {iteration}")
        elif self.database.best_program_id == child.id:
            logger.info(f"New best solution found at iteration {iteration}")

        # Checkpoint callback. In parallel mode the caller passes None and
        # checkpoints are fired from _mark_iteration_complete based on the
        # highest contiguous completed iteration, so we skip in-line
        # firing here to avoid duplicate / out-of-order checkpoints.
        if checkpoint_callback is not None and iteration > 0 and iteration % self.config.checkpoint_interval == 0:
            logger.info(f"Checkpoint interval reached at iteration {iteration}")
            self.database.log_status()
            checkpoint_callback(iteration)

    # =========================================================================
    # Child Generation
    # =========================================================================

    async def _generate_child(
        self,
        iteration: int,
        error_context: Optional[str] = None,
        force_exploration: bool = False,
        island_idx: Optional[int] = None,
    ) -> SerializableResult:
        """Generate and evaluate a single child program.

        Args:
            island_idx: Explicit island to sample from. When None (sequential
                mode) the database falls back to ``current_island``. Parallel
                schedulers must pass an island reserved via
                ``database.reserve_island()`` so concurrent in-flight
                iterations land on distinct islands.
        """
        try:
            if not self.database.programs:
                return await self._run_from_scratch_iteration(iteration)

            # Ensure all islands are seeded (needed after from-scratch bootstrap)
            self._ensure_all_islands_seeded()

            # Sample parent and context programs (database returns standard framework dicts)
            parent_dict, context_programs_dict = self.database.sample(
                self.num_context_programs,
                force_exploration=force_exploration,
                island_idx=island_idx,
            )

            # Unpack parent dict (standard framework pattern)
            if not parent_dict:
                logger.error("sample() returned empty parent dict")
                return SerializableResult(
                    error="Empty parent dict from sample()", iteration=iteration
                )
            parent_label = list(parent_dict.keys())[0]
            parent = list(parent_dict.values())[0]

            # Capture per-iteration sampling state immediately (sync, no
            # await) so parallel iterations don't overwrite each other's
            # values via shared controller/database fields.
            sampling_mode = getattr(self.database, "_last_sampling_mode", None) or "balanced"
            effective_island = island_idx if island_idx is not None else self.database.current_island
            if self.database.use_adaptive_search:
                sampling_intensity = self.database.adapter.get_search_intensity(effective_island)
            else:
                sampling_intensity = self.database.fixed_intensity

            # Sequential-mode compatibility: still stash on self for any
            # legacy reader. In parallel mode the values on the
            # SerializableResult are authoritative.
            self._last_sampling_mode = sampling_mode
            self._last_sampling_intensity = sampling_intensity

            # When paradigm is active, use best program as parent
            # This ensures paradigm (designed from best) is applied to best, not random parent
            paradigm = (
                self.database.get_current_paradigm()
                if self.database.use_paradigm_breakthrough
                else None
            )
            if paradigm:
                best_program = self.database.get_best_program()
                if best_program:
                    parent_dict = {parent_label: best_program}
                    parent = best_program
                    # Keep context_programs_dict from sampling for diversity

            # Gather siblings for sibling context
            siblings = []
            if hasattr(self.database, "get_children"):
                try:
                    siblings = self.database.get_children(parent.id)
                except (AttributeError, NotImplementedError):
                    pass

            # Build context for prompt generation
            # Only database-derived data — config values are read by the
            # context builder from self.config directly.
            context = {
                "program_metrics": parent.metrics,
                "other_context_programs": context_programs_dict,
                # AdaEvolve-specific keys (consumed by AdaEvolveContextBuilder)
                "paradigm": paradigm,
                "siblings": siblings,
                "error_context": error_context,
            }
            # Include any extra prompt context
            for k, v in self._prompt_context.items():
                if k not in context:
                    context[k] = v

            # Build prompt (AdaEvolveContextBuilder handles paradigm/sibling/error formatting)
            prompt = self.context_builder.build_prompt(parent_dict, context)

            # Mark paradigm as used after prompt is built
            if paradigm:
                self.database.use_paradigm()

            # Build tracking info for child program
            parent_info = (parent_label, parent.id)
            context_info = [
                (label, p.id) for label, programs in context_programs_dict.items() for p in programs
            ]
            context_program_ids = [
                p.id for programs in context_programs_dict.values() for p in programs
            ]

            # Apply human feedback (append or replace mode)
            if self.feedback_reader:
                self.feedback_reader.set_current_prompt(prompt["system"])
                feedback = self.feedback_reader.read()
                if feedback:
                    prompt = self.feedback_reader.apply_feedback(prompt)
                    self.feedback_reader.log_usage(iteration, feedback, self.feedback_reader.mode)

            # Generate and evaluate
            result = await self._execute_generation(
                parent,
                prompt,
                iteration,
                parent_info=parent_info,
                context_info=context_info,
                context_program_ids=context_program_ids,
                other_context_programs=context_programs_dict,
            )
            # Attach per-iteration scheduling state to the result so
            # parallel iterations can be logged correctly without sharing
            # mutable controller/database fields.
            result.sampling_mode = sampling_mode
            result.sampling_intensity = sampling_intensity
            result.island_idx = effective_island
            return result

        except Exception as e:
            logger.exception(f"Generation failed: {e}")
            return SerializableResult(error=str(e), iteration=iteration)

    # =========================================================================
    # LLM Generation
    # =========================================================================

    async def _execute_generation(
        self,
        parent: Program,
        prompt: Dict[str, str],
        iteration: int,
        parent_info: Optional[tuple] = None,
        context_info: Optional[List[tuple]] = None,
        context_program_ids: Optional[List[str]] = None,
        other_context_programs: Optional[Dict] = None,
    ) -> SerializableResult:
        """Execute LLM generation and evaluation."""
        start_time = time.time()

        image_path = None
        child_id = str(uuid.uuid4())

        # Generate
        llm_generation_time = 0.0
        try:
            llm_start = time.time()
            if self.config.language == "image":
                from skydiscover.search.utils.discovery_utils import build_image_content

                user_content = build_image_content(
                    prompt["user"], parent, other_context_programs or {}
                )
                result = await self._call_llm(
                    prompt["system"],
                    user_content,
                    image_output=True,
                    output_dir=self._get_image_output_dir(),
                    program_id=child_id,
                )
                response = result.text or ""
                image_path = result.image_path
                if not image_path:
                    return SerializableResult(
                        error="VLM did not generate an image", iteration=iteration
                    )
            else:
                result = await self._call_llm(prompt["system"], prompt["user"])
                response = result.text
            llm_generation_time = time.time() - llm_start
        except Exception as e:
            return SerializableResult(error=f"LLM error: {e}", iteration=iteration)

        if not response and self.config.language != "image":
            return SerializableResult(error="Empty LLM response", iteration=iteration)

        # Parse code from response
        if self.config.language == "image":
            child_solution = response or "(image generated)"
            changes = "Image generation"
        elif self.config.diff_based_generation:
            diffs = extract_diffs(response)
            if diffs:
                child_solution = apply_diff(parent.solution, response)
                changes = format_diff_summary(diffs)
            else:
                # No diffs found, try full rewrite
                child_solution = parse_full_rewrite(response, self.config.language)
                changes = "Full rewrite"
        else:
            child_solution = parse_full_rewrite(response, self.config.language)
            changes = "Full rewrite"

        if not child_solution:
            return SerializableResult(error="No valid solution in response", iteration=iteration)

        # Evaluate
        try:
            eval_input = image_path if self.config.language == "image" else child_solution
            eval_start = time.time()
            eval_result = await self.evaluator.evaluate_program(eval_input, child_id)
            eval_time = time.time() - eval_start
        except Exception as e:
            return SerializableResult(error=f"Evaluation error: {e}", iteration=iteration)

        metrics = eval_result.metrics
        artifacts = eval_result.artifacts

        # Extract image_path from evaluator metrics (non-image mode fallback)
        if not image_path:
            image_path = (
                metrics.pop("image_path", None)
                if isinstance(metrics.get("image_path"), str)
                else None
            )

        # Build child program with full tracking info
        child_metadata = {"changes": changes, "parent_metrics": parent.metrics}
        if image_path:
            child_metadata["image_path"] = image_path
        child = Program(
            id=child_id,
            solution=child_solution,
            language=self.config.language,
            metrics=metrics,
            iteration_found=iteration,
            parent_id=parent.id,
            other_context_ids=context_program_ids,
            parent_info=parent_info,
            context_info=context_info,
            generation=parent.generation + 1,
            metadata=child_metadata,
            artifacts=artifacts,
        )

        iteration_time = time.time() - start_time

        return SerializableResult(
            child_program_dict=child.to_dict(),
            parent_id=parent.id,
            other_context_ids=context_program_ids,
            iteration_time=iteration_time,
            llm_generation_time=llm_generation_time,
            eval_time=eval_time,
            prompt=prompt,
            llm_response=response,
            iteration=iteration,
        )

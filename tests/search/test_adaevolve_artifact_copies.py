"""Program copies (island seeding, spawned islands) keep the evaluator's artifacts.

Without them, a child bred from a copy got no evaluator feedback in its prompt.
"""

from skydiscover.config import AdaEvolveDatabaseConfig
from skydiscover.context_builder.utils import format_artifacts
from skydiscover.search.adaevolve.database import AdaEvolveDatabase
from skydiscover.search.base_database import Program

ARTIFACTS = {
    "test_n4_s1024": "16 GPUs (4 nodes x 4), 1 KiB: simulated 276,015 cycles",
    "fitness_explanation": "Fitness = reference / simulated.",
    "_evaluator_version": "2026-11-05.2",
}


def _seed():
    return Program(id="seed", solution="def f():\n    return 1\n",
                   metrics={"combined_score": 0.5}, artifacts=dict(ARTIFACTS))


def _db(num_islands, **extra):
    config = AdaEvolveDatabaseConfig(population_size=10, num_islands=num_islands,
                                     use_dynamic_islands=False, use_paradigm_breakthrough=False, **extra)
    return AdaEvolveDatabase("test", config)


def test_seed_all_islands_copies_artifacts():
    db = _db(num_islands=3)
    db.seed_all_islands(_seed(), iteration=0)
    copies = [p for p in db.programs.values() if p.id != "seed"]
    assert len(copies) == 2
    for p in copies:
        assert p.artifacts == ARTIFACTS
        assert p.artifacts is not ARTIFACTS  # an independent dict, not shared


def test_seed_new_island_copies_artifacts():
    db = _db(num_islands=2)
    db.seed_all_islands(_seed(), iteration=0)
    before = set(db.programs)
    db._seed_new_island(1)
    new = [db.programs[i] for i in set(db.programs) - before]
    assert new and all(p.artifacts == ARTIFACTS for p in new)


def test_ablation_flag_restores_metrics_only_copies():
    db = _db(num_islands=3, copies_keep_artifacts=False)
    db.seed_all_islands(_seed(), iteration=0)
    copies = [p for p in db.programs.values() if p.id != "seed"]
    assert len(copies) == 2 and all(p.artifacts == {} for p in copies)
    assert all(p.metrics == {"combined_score": 0.5} for p in copies)


def test_format_artifacts_hides_underscore_keys():
    text = format_artifacts(_seed())
    assert "## test_n4_s1024" in text and "## fitness_explanation" in text
    assert "_evaluator_version" not in text and "2026-11-05.2" not in text

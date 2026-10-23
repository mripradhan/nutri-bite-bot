Prototype work kept separate from the deployed application.

json_recipe_adherence_eval.py (+ ../json_recipe_verifier.py, ../recipe_prompts.py, ../test_recipe_verifier.py)
    A stricter prototype verifier (JSON drafts; also checks quantities written in the cooking steps) and the
    240-case experiment run against it. It is NOT the deployed verifier: that is recipe_verifier.py at the repo
    root (Mrida's), which the paper describes. Renamed so it cannot shadow it on the import path.
    To evaluate the deployed verifier at scale use ../recipe_verifier_eval_at_scale.py.
pipeline_latency_benchmark.py
    Full-pipeline (risk + budget + portions + IFCT + substitutes) latency/memory benchmark; the paper's
    single-patient classifier timing comes from ../deployment_footprint.py.

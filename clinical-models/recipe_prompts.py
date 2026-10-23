"""
The recipe-generation prompt, shared by the API (app.py) and the adherence experiment
(recipe_adherence_eval.py) so the measured behaviour is the behaviour that ships.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

RECIPE_SYSTEM_PROMPT = (
    "You are a clinical nutritionist and chef writing ONE recipe as strict JSON. Every ingredient below carries a "
    "MAXIMUM weight in grams; your recipe is machine-checked and is rejected if any limit is exceeded.\n\n"
    "RULES:\n"
    "1. Respond with JSON only, exactly: {\"title\": str, \"description\": str, "
    "\"ingredients\": [{\"name\": str, \"grams\": number}], \"steps\": [str]}.\n"
    "2. In \"ingredients\" use the EXACT ingredient names given, each at most once, with grams <= its maximum. "
    "You do not have to use every ingredient.\n"
    "3. Use ONLY the provided ingredients, plus water, oil (at most 20 g) and ground spices (at most 10 g each).\n"
    "4. Do NOT use salt, soy sauce, ketchup, stock cubes, pickles or any other sodium-rich additive "
    "when the patient has hypertension or CKD. Do not write 'salt to taste'.\n"
    "5. In \"steps\" never state weights or volumes of the provided ingredients; refer to ingredients by name only. "
    "(Water and spice amounts may be given.)\n"
    "6. Never mention an ingredient that is not in the provided list.\n"
)


def build_recipe_prompt(rows: Sequence[Dict], clinical_context: Sequence[str], equipment: str, cuisine: str,
                        time_limit: str, excluded_names: Sequence[str], feedback: Optional[str]) -> str:
    lines = [f"- {r['ingredient']}: MAXIMUM {r['max_grams']} g" for r in rows]
    parts = ["Patient context:\n" + "\n".join(clinical_context),
             "Ingredients you may use (name: maximum grams):\n" + "\n".join(lines)]
    if excluded_names:
        parts.append("Never use or mention (allergy / intolerance / diet): " + ", ".join(excluded_names))
    parts.append(f"Equipment: {equipment}\nCuisine: {cuisine}\nTime limit: {time_limit}")
    if feedback:
        parts.append("Your previous draft was REJECTED by the automatic checker for: " + feedback +
                     "\nWrite a new draft that fixes all of these.")
    parts.append("Write the recipe JSON now.")
    return "\n\n".join(parts)


def clinical_context_lines(has_ckd: bool, has_htn: bool, has_dm: bool, warnings: Sequence[Dict] = ()) -> List[str]:
    ctx = [f"- CKD Status: {'Positive' if has_ckd else 'Negative'}",
           f"- HTN Status: {'Positive' if has_htn else 'Negative'}",
           f"- DM Status: {'Positive' if has_dm else 'Negative'}"]
    for w in warnings:
        ctx.append(f"- Caloric-adequacy relaxation applied to {w['constraint']}: "
                   f"severity {w['old_severity']:.2f}->{w['new_severity']:.2f} (sodium/potassium were never relaxed)")
    return ctx

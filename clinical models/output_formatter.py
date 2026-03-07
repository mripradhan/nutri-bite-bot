"""
Pretty Output Formatter for Model2 Portion Control System
=========================================================
User-friendly display with clear visual formatting
"""

from typing import Dict, List, Any


class OutputFormatter:
    """Format Model2 output in a user-friendly way"""
    
    # Box drawing characters
    BOX_TL = "╭"
    BOX_TR = "╮"
    BOX_BL = "╰"
    BOX_BR = "╯"
    BOX_H = "─"
    BOX_V = "│"
    BOX_LT = "├"
    BOX_RT = "┤"
    
    # Status icons
    ICON_OK = "✅"
    ICON_WARN = "⚠️"
    ICON_STOP = "🚫"
    ICON_FOOD = "🍽️"
    ICON_PERSON = "👤"
    ICON_CHART = "📊"
    ICON_HEART = "💚"
    ICON_YELLOW = "💛"
    ICON_RED = "❤️"
    
    def __init__(self, width: int = 70):
        self.width = width
    
    def _box_top(self, title: str = "") -> str:
        """Create box top border with optional title"""
        if title:
            padding = self.width - len(title) - 4
            left_pad = padding // 2
            right_pad = padding - left_pad
            return f"{self.BOX_TL}{self.BOX_H * left_pad} {title} {self.BOX_H * right_pad}{self.BOX_TR}"
        return f"{self.BOX_TL}{self.BOX_H * (self.width - 2)}{self.BOX_TR}"
    
    def _box_bottom(self) -> str:
        return f"{self.BOX_BL}{self.BOX_H * (self.width - 2)}{self.BOX_BR}"
    
    def _box_divider(self) -> str:
        return f"{self.BOX_LT}{self.BOX_H * (self.width - 2)}{self.BOX_RT}"
    
    def _box_line(self, text: str, padding: int = 2) -> str:
        """Create a line inside a box"""
        content_width = self.width - 4 - (padding * 2)
        if len(text) > content_width:
            text = text[:content_width-3] + "..."
        return f"{self.BOX_V}{' ' * (padding + 1)}{text.ljust(content_width + padding + 1)}{self.BOX_V}"
    
    def _risk_badge(self, risk) -> str:
        """Create a risk level badge"""
        # Handle enriched dict format: {"label": "high", "severity_score": ...}
        if isinstance(risk, dict):
            risk = risk.get("label", "moderate")
        risk = risk.lower()
        if risk == "low":
            return f"{self.ICON_HEART} LOW"
        elif risk == "moderate":
            return f"{self.ICON_YELLOW} MODERATE"
        else:
            return f"{self.ICON_RED} HIGH"
    
    def _portion_badge(self, label: str, grams: float) -> str:
        """Create portion recommendation badge"""
        if label == "Allowed":
            return f"{self.ICON_OK} {grams:.0f}g"
        elif label == "Half portion":
            return f"{self.ICON_WARN} {grams:.0f}g (reduced)"
        else:
            return f"{self.ICON_STOP} AVOID"
    
    def format_recommendations(self, results: Dict[str, Any]) -> str:
        """Format complete recommendation results"""
        lines = []
        
        # Header
        lines.append("")
        lines.append(self._box_top("NUTRI-BITE PORTION ADVISOR"))
        lines.append(self._box_line(""))
        
        # Patient Conditions
        conditions = results.get("patient_conditions", {})
        active_conditions = [k.replace("has_", "").upper() for k, v in conditions.items() if v]
        condition_text = ", ".join(active_conditions) if active_conditions else "No chronic conditions"
        lines.append(self._box_line(f"{self.ICON_PERSON} Patient: {condition_text}"))
        lines.append(self._box_line(""))
        lines.append(self._box_divider())
        
        # Risk Levels Section
        lines.append(self._box_line(""))
        lines.append(self._box_line(f"{self.ICON_CHART} CLINICAL RISK ASSESSMENT"))
        lines.append(self._box_line(""))
        
        risk_levels = results.get("risk_levels", {})
        risk_display = {
            "sodium_sensitivity": "Salt/Sodium",
            "potassium_sensitivity": "Potassium", 
            "protein_restriction": "Protein",
            "carb_sensitivity": "Carbohydrates",
            "phosphorus_sensitivity": "Phosphorus"
        }
        
        for key, display_name in risk_display.items():
            risk = risk_levels.get(key, "moderate")
            badge = self._risk_badge(risk)
            lines.append(self._box_line(f"   {display_name:20} {badge}"))
        
        lines.append(self._box_line(""))
        lines.append(self._box_divider())
        
        # Daily Budget
        lines.append(self._box_line(""))
        lines.append(self._box_line(f"{self.ICON_FOOD} YOUR DAILY LIMITS (remaining)"))
        lines.append(self._box_line(""))
        
        budget = results.get("daily_budget", {})
        budget_display = [
            ("Sodium", budget.get("sodium_mg", 0), "mg"),
            ("Potassium", budget.get("potassium_mg", 0), "mg"),
            ("Protein", budget.get("protein_g", 0), "g"),
            ("Carbohydrates", budget.get("carbs_g", 0), "g"),
            ("Phosphorus", budget.get("phosphorus_mg", 0), "mg"),
        ]
        
        for name, value, unit in budget_display:
            bar_total = 20
            # Assume typical max values for bar display
            max_vals = {"Sodium": 2300, "Potassium": 4700, "Protein": 56, "Carbohydrates": 275, "Phosphorus": 1250}
            max_val = max_vals.get(name, 100)
            filled = int((value / max_val) * bar_total) if max_val > 0 else 0
            filled = min(filled, bar_total)
            bar = "█" * filled + "░" * (bar_total - filled)
            lines.append(self._box_line(f"   {name:15} [{bar}] {value:.0f}{unit}"))
        
        lines.append(self._box_line(""))
        lines.append(self._box_divider())
        
        # Recommendations
        lines.append(self._box_line(""))
        lines.append(self._box_line(f"🥗 PORTION RECOMMENDATIONS"))
        lines.append(self._box_line(""))
        
        recommendations = results.get("recommendations", [])
        
        # Sort by label: Allowed first, then Half portion, then Avoid
        label_order = {"Allowed": 0, "Half portion": 1, "Avoid": 2}
        sorted_recs = sorted(recommendations, key=lambda r: label_order.get(r["label"], 3))
        
        for rec in sorted_recs:
            ingredient = rec["ingredient"]
            label = rec["label"]
            grams = rec["max_grams"]
            constraint = rec.get("binding_constraint", "")
            
            badge = self._portion_badge(label, grams)
            
            # Shorten ingredient name if needed
            if len(ingredient) > 25:
                ingredient = ingredient[:22] + "..."
            
            lines.append(self._box_line(f"   {ingredient:25} {badge}"))
            
            # Add constraint note for restricted items
            if label != "Allowed" and constraint:
                constraint_names = {
                    "sodium": "sodium content",
                    "potassium": "potassium content", 
                    "protein": "protein content",
                    "carbs": "carb content"
                }
                note = constraint_names.get(constraint, constraint)
                lines.append(self._box_line(f"   {'':25} ↳ Limited by {note}"))
        
        lines.append(self._box_line(""))
        lines.append(self._box_divider())
        
        # Summary Section
        lines.append(self._box_line(""))
        lines.append(self._box_line("📋 QUICK SUMMARY"))
        lines.append(self._box_line(""))
        
        summary = results.get("summary", {})
        allowed = summary.get("allowed", [])
        half_portion = summary.get("half_portion", [])
        avoid = summary.get("avoid", [])
        
        if allowed:
            allowed_names = [a.split("(")[0].strip()[:15] for a in allowed[:4]]
            if len(allowed) > 4:
                allowed_names.append(f"+{len(allowed)-4} more")
            lines.append(self._box_line(f"   {self.ICON_OK} Full portion OK: {', '.join(allowed_names)}"))
        
        if half_portion:
            hp_names = [h.split("(")[0].strip()[:15] for h in half_portion[:4]]
            if len(half_portion) > 4:
                hp_names.append(f"+{len(half_portion)-4} more")
            lines.append(self._box_line(f"   {self.ICON_WARN} Use half portion: {', '.join(hp_names)}"))
        
        if avoid:
            avoid_names = [a.split("(")[0].strip()[:15] for a in avoid[:4]]
            if len(avoid) > 4:
                avoid_names.append(f"+{len(avoid)-4} more")
            lines.append(self._box_line(f"   {self.ICON_STOP} Best to avoid: {', '.join(avoid_names)}"))
        
        if not allowed and not half_portion and not avoid:
            lines.append(self._box_line("   No recommendations generated"))
        
        lines.append(self._box_line(""))
        lines.append(self._box_bottom())
        lines.append("")
        
        return "\n".join(lines)
    
    def format_simple(self, results: Dict[str, Any]) -> str:
        """Format a simplified one-line summary for each ingredient"""
        lines = []
        lines.append("")
        lines.append("=" * 60)
        lines.append("  PORTION RECOMMENDATIONS")
        lines.append("=" * 60)
        
        for rec in results.get("recommendations", []):
            icon = self.ICON_OK if rec["label"] == "Allowed" else (self.ICON_WARN if rec["label"] == "Half portion" else self.ICON_STOP)
            lines.append(f"  {icon} {rec['ingredient']}: {rec['max_grams']:.0f}g ({rec['label']})")
        
        lines.append("=" * 60)
        return "\n".join(lines)


def print_recommendations(results: Dict[str, Any], simple: bool = False):
    """Print formatted recommendations"""
    formatter = OutputFormatter()
    if simple:
        print(formatter.format_simple(results))
    else:
        print(formatter.format_recommendations(results))


# Demo with sample data
if __name__ == "__main__":
    # Sample results for testing
    sample_results = {
        "patient_conditions": {
            "has_ckd": True,
            "has_htn": True,
            "has_dm": True,
        },
        "risk_levels": {
            "sodium_sensitivity": "high",
            "potassium_sensitivity": "high",
            "protein_restriction": "high",
            "carb_sensitivity": "high",
        },
        "daily_budget": {
            "sodium_mg": 1500,
            "potassium_mg": 2000,
            "protein_g": 42,
            "carbs_g": 150,
        },
        "recommendations": [
            {"ingredient": "Cucumber", "max_grams": 143.9, "label": "Allowed", "binding_constraint": "potassium"},
            {"ingredient": "Apple", "max_grams": 114.4, "label": "Allowed", "binding_constraint": "carbs"},
            {"ingredient": "Banana, ripe", "max_grams": 55.9, "label": "Half portion", "binding_constraint": "potassium"},
            {"ingredient": "Spinach (Palak)", "max_grams": 35.8, "label": "Half portion", "binding_constraint": "potassium"},
            {"ingredient": "Potato (Aloo)", "max_grams": 46.9, "label": "Half portion", "binding_constraint": "potassium"},
            {"ingredient": "Rice, milled (white)", "max_grams": 19.2, "label": "Half portion", "binding_constraint": "carbs"},
            {"ingredient": "Prawn (Jhinga)", "max_grams": 22.1, "label": "Half portion", "binding_constraint": "protein"},
        ],
        "summary": {
            "allowed": ["Cucumber", "Apple"],
            "half_portion": ["Banana, ripe", "Spinach (Palak)", "Potato (Aloo)", "Rice, milled (white)", "Prawn (Jhinga)"],
            "avoid": [],
        }
    }
    
    print_recommendations(sample_results)

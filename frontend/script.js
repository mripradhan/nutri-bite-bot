/**
 * NutriBiteBot — Frontend Logic
 * ==============================
 * Handles form submission, API calls to Flask backend,
 * and dynamic rendering of ML model predictions.
 *
 * ALL predictions come from the backend's model.predict() /
 * model.predict_proba() — no client-side heuristics.
 */

// ─── API Base URL ──────────────────────────────────────────
const API_BASE = "https://nutri-bite-bot.onrender.com";

// ─── Preset Patient Profiles ───────────────────────────────
const PRESETS = {
    healthy: {
        age: 35, sex_male: 0, has_htn: false, has_dm: false, has_ckd: false,
        serum_sodium: 140, serum_potassium: 4.0, creatinine: 0.9,
        egfr: 110, hba1c: 5.2, fbs: 88, sbp: 118, dbp: 75, bmi: 22,
    },
    ckd: {
        age: 62, sex_male: 1, has_htn: false, has_dm: false, has_ckd: true,
        serum_sodium: 138, serum_potassium: 5.8, creatinine: 3.5,
        egfr: 22, hba1c: 5.5, fbs: 95, sbp: 135, dbp: 82, bmi: 26,
    },
    diabetes: {
        age: 48, sex_male: 1, has_htn: false, has_dm: true, has_ckd: false,
        serum_sodium: 140, serum_potassium: 4.0, creatinine: 1.0,
        egfr: 92, hba1c: 9.2, fbs: 210, sbp: 125, dbp: 80, bmi: 32,
    },
    htn: {
        age: 58, sex_male: 0, has_htn: true, has_dm: false, has_ckd: false,
        serum_sodium: 143, serum_potassium: 3.8, creatinine: 1.1,
        egfr: 78, hba1c: 5.6, fbs: 102, sbp: 165, dbp: 98, bmi: 29,
    },
    multi: {
        age: 65, sex_male: 1, has_htn: true, has_dm: true, has_ckd: true,
        serum_sodium: 145, serum_potassium: 5.9, creatinine: 4.0,
        egfr: 18, hba1c: 8.8, fbs: 190, sbp: 170, dbp: 100, bmi: 34,
    },
};

// ─── Initialization ────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    initPresets();
    initForm();
    initModelInfo();
});

// ─── Preset Buttons ────────────────────────────────────────
function initPresets() {
    document.querySelectorAll(".preset-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            const preset = PRESETS[btn.dataset.preset];
            if (!preset) return;

            // Remove active from all, add to clicked
            document.querySelectorAll(".preset-btn").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");

            fillForm(preset);
        });
    });
}

function fillForm(data) {
    // Numeric fields
    const numericFields = [
        "age", "bmi", "serum_sodium", "serum_potassium", "creatinine",
        "egfr", "hba1c", "fbs", "sbp", "dbp",
    ];
    numericFields.forEach(name => {
        const el = document.getElementById(name);
        if (el && data[name] !== undefined) el.value = data[name];
    });

    // Sex select
    const sexEl = document.getElementById("sex_male");
    if (sexEl && data.sex_male !== undefined) {
        sexEl.value = data.sex_male ? "1" : "0";
    }

    // Condition toggles
    ["has_htn", "has_dm", "has_ckd"].forEach(name => {
        const el = document.getElementById(name);
        if (el) el.checked = !!data[name];
    });
}

// ─── Form Submission ───────────────────────────────────────
function initForm() {
    const form = document.getElementById("patient-form");
    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        await runPrediction();
    });
}

async function runPrediction() {
    const btn = document.getElementById("analyze-btn");
    const btnText = btn.querySelector(".btn-text");
    const btnLoader = btn.querySelector(".btn-loader");

    // Show loading state
    btn.disabled = true;
    btnText.style.display = "none";
    btnLoader.style.display = "inline";

    // Clear any previous error
    document.querySelectorAll(".error-banner").forEach(el => el.remove());

    try {
        // Build payload from form (using the EXACT feature names the model expects)
        const payload = {
            age: parseFloat(document.getElementById("age").value),
            sex_male: parseInt(document.getElementById("sex_male").value),
            has_htn: document.getElementById("has_htn").checked ? 1 : 0,
            has_dm: document.getElementById("has_dm").checked ? 1 : 0,
            has_ckd: document.getElementById("has_ckd").checked ? 1 : 0,
            serum_sodium: parseFloat(document.getElementById("serum_sodium").value),
            serum_potassium: parseFloat(document.getElementById("serum_potassium").value),
            creatinine: parseFloat(document.getElementById("creatinine").value),
            egfr: parseFloat(document.getElementById("egfr").value),
            hba1c: parseFloat(document.getElementById("hba1c").value),
            fbs: parseFloat(document.getElementById("fbs").value),
            sbp: parseFloat(document.getElementById("sbp").value),
            dbp: parseFloat(document.getElementById("dbp").value),
            bmi: parseFloat(document.getElementById("bmi").value),
        };

        // Validate no NaN
        for (const [key, val] of Object.entries(payload)) {
            if (isNaN(val)) {
                throw new Error(`Invalid value for ${key}`);
            }
        }

        // Call Flask backend — model.predict() runs server-side
        const res = await fetch(`${API_BASE}/api/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || `Server error: ${res.status}`);
        }

        const data = await res.json();
        renderResults(data);

    } catch (err) {
        showError(err.message);
    } finally {
        btn.disabled = false;
        btnText.style.display = "inline";
        btnLoader.style.display = "none";
    }
}

// ─── Results Rendering ─────────────────────────────────────
function renderResults(data) {
    const section = document.getElementById("results-section");
    section.style.display = "";

    renderPatientSummary(data.patient_summary);
    renderRiskGrid(data.risk_levels);
    renderProbabilityCharts(data.risk_levels);

    // Smooth scroll to results
    section.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderPatientSummary(summary) {
    const container = document.getElementById("summary-content");
    const labs = summary.key_labs;
    const conditions = summary.conditions;

    let html = `<div class="summary-grid">`;
    html += summaryItem("Age", summary.age + " years");
    html += summaryItem("Sex", summary.sex);

    for (const [name, val] of Object.entries(labs)) {
        html += summaryItem(name, val);
    }
    html += `</div>`;

    // Condition tags
    html += `<div class="condition-tags">`;
    for (const [name, active] of Object.entries(conditions)) {
        html += `<span class="condition-tag ${active ? 'active' : 'inactive'}">${name}: ${active ? 'Yes' : 'No'}</span>`;
    }
    html += `</div>`;

    container.innerHTML = html;
}

function summaryItem(label, value) {
    return `
        <div class="summary-item">
            <div class="label">${label}</div>
            <div class="value">${value}</div>
        </div>
    `;
}

function renderRiskGrid(riskLevels) {
    const grid = document.getElementById("risk-grid");
    let html = "";

    for (const [target, info] of Object.entries(riskLevels)) {
        html += `
            <div class="risk-card risk-${info.label}">
                <div class="risk-card-header">
                    <span class="risk-card-title">${info.display_name}</span>
                    <span class="risk-badge ${info.label}">${info.label}</span>
                </div>

                <div class="risk-confidence">
                    <div class="confidence-bar-track">
                        <div class="confidence-bar-fill ${info.label}"
                             style="width: 0%"
                             data-width="${info.confidence}%"></div>
                    </div>
                    <div class="confidence-label">
                        <span>Model Confidence</span>
                        <span class="confidence-value">${info.confidence}%</span>
                    </div>
                </div>

                <div class="risk-note">${info.clinical_note}</div>
            </div>
        `;
    }

    grid.innerHTML = html;

    // Animate confidence bars
    requestAnimationFrame(() => {
        grid.querySelectorAll(".confidence-bar-fill").forEach(bar => {
            bar.style.width = bar.dataset.width;
        });
    });
}

function renderProbabilityCharts(riskLevels) {
    const container = document.getElementById("probability-charts");
    let html = `<div class="prob-grid">`;

    for (const [target, info] of Object.entries(riskLevels)) {
        const probs = info.probabilities;
        html += `
            <div class="prob-card">
                <div class="prob-card-title">${info.display_name}</div>
                ${probRow("low", probs.low)}
                ${probRow("moderate", probs.moderate)}
                ${probRow("high", probs.high)}
            </div>
        `;
    }

    html += `</div>`;
    container.innerHTML = html;

    // Animate probability bars
    requestAnimationFrame(() => {
        container.querySelectorAll(".prob-bar-fill").forEach(bar => {
            bar.style.width = bar.dataset.width;
        });
    });
}

function probRow(level, value) {
    return `
        <div class="prob-row">
            <span class="prob-label ${level}">${level}</span>
            <div class="prob-bar-track">
                <div class="prob-bar-fill ${level}"
                     style="width: 0%"
                     data-width="${value}%"></div>
            </div>
            <span class="prob-value">${value.toFixed(1)}%</span>
        </div>
    `;
}


function formatNutrientName(key) {
    return key
        .replace(/_/g, " ")
        .replace(/\b\w/g, c => c.toUpperCase())
        .replace("Mg", "(mg)")
        .replace("Ml", "(mL)")
        .replace(" G", " (g)");
}

// ─── Model Info ────────────────────────────────────────────
function initModelInfo() {
    const toggleBtn = document.getElementById("toggle-model-info");
    const body = document.getElementById("model-info-body");
    let loaded = false;

    toggleBtn.addEventListener("click", async () => {
        const isHidden = body.style.display === "none";
        body.style.display = isHidden ? "" : "none";
        toggleBtn.textContent = isHidden ? "Hide Details" : "Show Details";

        if (!loaded) {
            loaded = true;
            await loadModelInfo();
        }
    });
}

async function loadModelInfo() {
    const container = document.getElementById("model-info-content");

    try {
        const res = await fetch(`${API_BASE}/api/model-info`);
        const data = await res.json();

        let html = `<div class="model-detail-grid">`;

        // Accuracy metrics
        html += `
            <div class="model-detail-card">
                <h4>Accuracy Metrics</h4>
                <div class="metric-row">
                    <span class="metric-name">Mean Accuracy</span>
                    <span class="metric-value">${(data.accuracy_metrics.mean_accuracy * 100).toFixed(2)}%</span>
                </div>
                <div class="metric-row">
                    <span class="metric-name">Mean F1 (weighted)</span>
                    <span class="metric-value">${(data.accuracy_metrics.mean_f1_weighted * 100).toFixed(2)}%</span>
                </div>
                <div class="metric-row">
                    <span class="metric-name">Cohen's Kappa</span>
                    <span class="metric-value">${data.accuracy_metrics.mean_cohen_kappa.toFixed(4)}</span>
                </div>
            </div>
        `;

        // Models
        for (const [target, info] of Object.entries(data.models)) {
            html += `
                <div class="model-detail-card">
                    <h4>${info.display_name}</h4>
                    <p>Type: <strong>${info.type}</strong></p>
                    <p>Classes: ${info.classes.map(c => `<span class="risk-badge ${c}" style="margin-left:0.2rem;">${c}</span>`).join(" ")}</p>
                </div>
            `;
        }

        // Features
        html += `
            <div class="model-detail-card">
                <h4>Input Features (${data.feature_count})</h4>
                <p>Preprocessing: ${data.preprocessing.join(" → ")}</p>
                <div class="feature-tags">
                    ${data.feature_names.map(f => `<span class="feature-tag">${f}</span>`).join("")}
                </div>
            </div>
        `;

        html += `</div>`;
        container.innerHTML = html;

    } catch (err) {
        container.innerHTML = `<p style="color:var(--risk-high);">Failed to load model info: ${err.message}</p>`;
    }
}

// ─── Error Display ─────────────────────────────────────────
function showError(message) {
    const existing = document.querySelectorAll(".error-banner");
    existing.forEach(el => el.remove());

    const banner = document.createElement("div");
    banner.className = "error-banner";
    banner.textContent = `Error: ${message}`;

    const form = document.getElementById("patient-form");
    form.parentNode.insertBefore(banner, form);
}


// ══════════════════════════════════════════════════════════════════
//  INGREDIENT PORTION RECOMMENDATION (MODEL 2)
// ══════════════════════════════════════════════════════════════════

const selectedIngredients = [];
let allIngredientsByCategory = null;  // cached from /api/ingredients
let searchTimeout = null;

document.addEventListener("DOMContentLoaded", () => {
    initIngredientSearch();
    initRecommendButton();
});

// ─── Autocomplete Search ───────────────────────────────────
function initIngredientSearch() {
    const input = document.getElementById("ingredient-search");
    const dropdown = document.getElementById("autocomplete-dropdown");
    const browseBtn = document.getElementById("btn-browse-all");

    input.addEventListener("input", () => {
        clearTimeout(searchTimeout);
        const query = input.value.trim();
        if (query.length < 2) {
            dropdown.style.display = "none";
            return;
        }
        searchTimeout = setTimeout(() => searchIngredients(query), 200);
    });

    input.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
            dropdown.style.display = "none";
        }
    });

    // Click outside to close
    document.addEventListener("click", (e) => {
        if (!e.target.closest(".search-input-wrap")) {
            dropdown.style.display = "none";
        }
    });

    // Browse all
    browseBtn.addEventListener("click", async () => {
        if (!allIngredientsByCategory) {
            try {
                const res = await fetch(`${API_BASE}/api/ingredients`);
                const data = await res.json();
                allIngredientsByCategory = data.by_category;
            } catch { return; }
        }
        showBrowseAll();
    });
}

async function searchIngredients(query) {
    const dropdown = document.getElementById("autocomplete-dropdown");
    try {
        const res = await fetch(`${API_BASE}/api/ingredients?q=${encodeURIComponent(query)}`);
        const data = await res.json();
        renderAutocomplete(data.ingredients);
    } catch {
        dropdown.style.display = "none";
    }
}

function renderAutocomplete(items) {
    const dropdown = document.getElementById("autocomplete-dropdown");
    if (!items.length) {
        dropdown.innerHTML = `<div class="autocomplete-item" style="color:var(--text-muted);">No matches found</div>`;
        dropdown.style.display = "";
        return;
    }

    dropdown.innerHTML = items.map(name => {
        const already = selectedIngredients.includes(name);
        return `
            <div class="autocomplete-item ${already ? 'active' : ''}"
                 data-name="${name}"
                 onclick="addIngredient('${name.replace(/'/g, "\\'")}')">
                <span>${name}</span>
                ${already ? '<span style="font-size:0.7rem;">&#10003;</span>' : ''}
            </div>
        `;
    }).join("");
    dropdown.style.display = "";
}

function showBrowseAll() {
    const dropdown = document.getElementById("autocomplete-dropdown");
    let html = "";
    for (const [cat, items] of Object.entries(allIngredientsByCategory)) {
        html += `<div class="autocomplete-group-label">${cat}</div>`;
        html += items.map(name => {
            const already = selectedIngredients.includes(name);
            return `
                <div class="autocomplete-item ${already ? 'active' : ''}"
                     data-name="${name}"
                     onclick="addIngredient('${name.replace(/'/g, "\\'")}')">
                    <span>${name}</span>
                    ${already ? '<span style="font-size:0.7rem;">&#10003;</span>' : ''}
                </div>
            `;
        }).join("");
    }
    dropdown.innerHTML = html;
    dropdown.style.display = "";
}

// ─── Ingredient Selection ──────────────────────────────────
function addIngredient(name) {
    if (selectedIngredients.includes(name)) return;
    selectedIngredients.push(name);
    renderSelectedIngredients();
    updateRecommendButton();
    // Clear search
    document.getElementById("ingredient-search").value = "";
    document.getElementById("autocomplete-dropdown").style.display = "none";
}

function removeIngredient(name) {
    const idx = selectedIngredients.indexOf(name);
    if (idx >= 0) selectedIngredients.splice(idx, 1);
    renderSelectedIngredients();
    updateRecommendButton();
}

function renderSelectedIngredients() {
    const container = document.getElementById("selected-ingredients");
    const empty = document.getElementById("empty-ingredients");

    if (!selectedIngredients.length) {
        container.innerHTML = `<p class="empty-state" id="empty-ingredients">No ingredients selected. Search above to add items.</p>`;
        return;
    }

    container.innerHTML = selectedIngredients.map(name => `
        <span class="ingredient-chip">
            ${name}
            <button class="chip-remove" onclick="removeIngredient('${name.replace(/'/g, "\\'")}')" title="Remove">x</button>
        </span>
    `).join("");
}

function updateRecommendButton() {
    document.getElementById("recommend-btn").disabled = selectedIngredients.length === 0;
}

// ─── Recommendation Request ────────────────────────────────
function initRecommendButton() {
    const btn = document.getElementById("recommend-btn");
    btn.addEventListener("click", runRecommendation);
}

async function runRecommendation() {
    const btn = document.getElementById("recommend-btn");
    const btnText = btn.querySelector(".btn-text");
    const btnLoader = btn.querySelector(".btn-loader");

    btn.disabled = true;
    btnText.style.display = "none";
    btnLoader.style.display = "inline";

    try {
        // Gather patient data from form
        const patient = {
            age: parseFloat(document.getElementById("age").value),
            sex_male: parseInt(document.getElementById("sex_male").value),
            has_htn: document.getElementById("has_htn").checked ? 1 : 0,
            has_dm: document.getElementById("has_dm").checked ? 1 : 0,
            has_ckd: document.getElementById("has_ckd").checked ? 1 : 0,
            serum_sodium: parseFloat(document.getElementById("serum_sodium").value),
            serum_potassium: parseFloat(document.getElementById("serum_potassium").value),
            creatinine: parseFloat(document.getElementById("creatinine").value),
            egfr: parseFloat(document.getElementById("egfr").value),
            hba1c: parseFloat(document.getElementById("hba1c").value),
            fbs: parseFloat(document.getElementById("fbs").value),
            sbp: parseFloat(document.getElementById("sbp").value),
            dbp: parseFloat(document.getElementById("dbp").value),
            bmi: parseFloat(document.getElementById("bmi").value),
        };

        // Validate
        for (const [key, val] of Object.entries(patient)) {
            if (isNaN(val)) throw new Error(`Fill in patient data first (missing: ${key})`);
        }

        const res = await fetch(`${API_BASE}/api/recommend`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ patient, ingredients: selectedIngredients }),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || `Server error: ${res.status}`);
        }

        const data = await res.json();
        renderRecommendations(data);

    } catch (err) {
        showError(err.message);
    } finally {
        btn.disabled = selectedIngredients.length === 0;
        btnText.style.display = "inline";
        btnLoader.style.display = "none";
    }
}

// ─── Render Recommendations ────────────────────────────────
function renderRecommendations(data) {
    const section = document.getElementById("recommend-results");
    section.style.display = "";

    renderBudget(data.daily_budget);
    renderRecCards(data.recommendations);

    section.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderBudget(budget) {
    const container = document.getElementById("budget-section");
    const items = [
        { label: "Sodium", value: `${budget.sodium_mg} mg`, key: "sodium_mg" },
        { label: "Potassium", value: `${budget.potassium_mg} mg`, key: "potassium_mg" },
        { label: "Protein", value: `${budget.protein_g} g`, key: "protein_g" },
        { label: "Carbs", value: `${budget.carbs_g} g`, key: "carbs_g" },
        { label: "Phosphorus", value: `${budget.phosphorus_mg} mg`, key: "phosphorus_mg" },
    ];

    container.innerHTML = `
        <p style="font-size:0.75rem; color:var(--text-muted); margin-bottom:0.5rem; font-weight:600; text-transform:uppercase; letter-spacing:0.04em;">
            Daily Nutrient Budget
        </p>
        <div class="budget-grid">
            ${items.map(i => `
                <div class="budget-item">
                    <div class="b-label">${i.label}</div>
                    <div class="b-value">${i.value}</div>
                </div>
            `).join("")}
        </div>
    `;
}

function renderRecCards(recommendations) {
    const grid = document.getElementById("recommend-grid");

    grid.innerHTML = recommendations.map(rec => {
        const labelClass = rec.label === "Allowed" ? "allowed"
            : rec.label === "Half Portion" ? "half"
                : rec.label === "Avoid" ? "avoid" : "notfound";

        const cardClass = rec.label === "Allowed" ? "rec-allowed"
            : rec.label === "Half Portion" ? "rec-half"
                : rec.label === "Avoid" ? "rec-avoid" : "rec-notfound";

        const load = rec.nutrient_load || {};

        return `
            <div class="rec-card ${cardClass}">
                <div class="rec-info">
                    <div class="rec-header">
                        <span class="rec-name">${rec.ingredient}</span>
                        ${rec.category ? `<span class="rec-category-tag">${rec.category}</span>` : ""}
                        <span class="rec-label ${labelClass}">${rec.label}</span>
                    </div>

                    ${rec.explanation ? `<div class="rec-detail">${rec.explanation}</div>` : ""}

                    ${Object.keys(load).length ? `
                    <div class="rec-nutrients">
                        ${load.sodium_mg !== undefined ? `<span class="rec-nut-item">Na: <span>${load.sodium_mg}mg</span></span>` : ""}
                        ${load.potassium_mg !== undefined ? `<span class="rec-nut-item">K: <span>${load.potassium_mg}mg</span></span>` : ""}
                        ${load.protein_g !== undefined ? `<span class="rec-nut-item">Prot: <span>${load.protein_g}g</span></span>` : ""}
                        ${load.carbs_g !== undefined ? `<span class="rec-nut-item">Carb: <span>${load.carbs_g}g</span></span>` : ""}
                        ${load.calories !== undefined ? `<span class="rec-nut-item">Cal: <span>${load.calories}</span></span>` : ""}
                    </div>` : ""}

                    ${rec.binding_constraint && rec.binding_constraint !== "unknown" ? `
                    <div class="rec-binding">Limiting factor: ${rec.binding_constraint}</div>` : ""}
                </div>

                <div class="rec-grams">
                    <div class="rec-grams-value ${labelClass}">${rec.max_grams}</div>
                    <div class="rec-grams-unit">grams max</div>
                </div>
            </div>
        `;
    }).join("");
}

// ══════════════════════════════════════════════════════════════════
//  PHASE 3: FRIDGE SCANNER & RECIPE GENERATOR
// ══════════════════════════════════════════════════════════════════

let detectedMapppedIngredients = [];

document.addEventListener("DOMContentLoaded", () => {
    initFridgeScanner();
    initRecipeGenerator();
});

function initFridgeScanner() {
    const dropzone = document.getElementById("fridge-dropzone");
    const imageInput = document.getElementById("fridge-image-input");

    if (!dropzone) return;

    dropzone.addEventListener("click", () => imageInput.click());

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("drag-over");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("drag-over");
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("drag-over");
        if (e.dataTransfer.files.length) {
            handleFridgeImage(e.dataTransfer.files[0]);
        }
    });

    imageInput.addEventListener("change", () => {
        if (imageInput.files.length) {
            handleFridgeImage(imageInput.files[0]);
        }
    });

    document.getElementById("btn-auto-add").addEventListener("click", () => {
        detectedMapppedIngredients.forEach(ing => addIngredient(ing));
        showError("Ingredients added! Scroll down to Portion Analysis, wait for it to run, then click 'Get Portion Recommendations' before generating a recipe.", true);

        // Scroll to portion analysis
        document.getElementById("ingredient-section").scrollIntoView({ behavior: "smooth" });
    });
}

function showError(message, isInfo = false) {
    const existing = document.querySelectorAll(".error-banner");
    existing.forEach(el => el.remove());

    const banner = document.createElement("div");
    banner.className = "error-banner";
    if (isInfo) {
        banner.style.background = "var(--risk-low-bg)";
        banner.style.borderColor = "var(--risk-low)";
        banner.style.color = "var(--risk-low)";
        banner.textContent = message;
    } else {
        banner.textContent = `Error: ${message}`;
    }

    const form = document.getElementById("patient-form");
    form.parentNode.insertBefore(banner, form);
}

async function handleFridgeImage(file) {
    // Show preview
    const preview = document.getElementById("preview-image");
    const content = document.getElementById("dropzone-content");

    preview.src = URL.createObjectURL(file);
    preview.classList.remove("hidden");
    content.classList.add("hidden");

    // Upload & Detect
    const form = new FormData();
    form.append("image", file);

    const badgesContainer = document.getElementById("detected-badges");
    badgesContainer.innerHTML = `<span class="detection-badge">Scanning image...</span>`;
    document.getElementById("detection-results").style.display = "";

    try {
        const res = await fetch(`${API_BASE}/api/detect`, {
            method: "POST",
            body: form
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || "Failed to detect ingredients");
        }

        const data = await res.json();

        if (data.mapped_ifct.length === 0) {
            badgesContainer.innerHTML = `<span class="detection-badge" style="color:var(--risk-high)">No matching ingredients found.</span>`;
            detectedMapppedIngredients = [];
            return;
        }

        detectedMapppedIngredients = data.mapped_ifct;
        badgesContainer.innerHTML = data.mapped_ifct.map(ing => `
            <span class="detection-badge">
                ${ing} <span class="match-icon">&#10003;</span>
            </span>
        `).join("");

    } catch (err) {
        showError(err.message);
        badgesContainer.innerHTML = `<span class="detection-badge" style="color:var(--risk-high)">Detection failed.</span>`;
    }
}

// ── Recipe Generator ──

function initRecipeGenerator() {
    const btn = document.getElementById("btn-generate-recipe");
    if (!btn) return;

    btn.addEventListener("click", generateClinicalRecipe);

    // Watch for recommend results to enable recipe button
    const observer = new MutationObserver(() => {
        const recommendResults = document.getElementById("recommend-results");
        btn.disabled = recommendResults.style.display === "none";
    });

    const resultsNode = document.getElementById("recommend-results");
    if (resultsNode) {
        observer.observe(resultsNode, { attributes: true, attributeFilter: ['style'] });
    }
}

async function generateClinicalRecipe() {
    const btn = document.getElementById("btn-generate-recipe");
    const btnLoader = btn.querySelector(".btn-loader");
    const btnText = btn.querySelector(".btn-text");
    const outputSection = document.getElementById("recipe-output");
    const contentDiv = document.getElementById("recipe-content");

    if (selectedIngredients.length === 0) {
        showError("No ingredients selected for the recipe");
        return;
    }

    btn.disabled = true;
    btnText.style.display = "none";
    btnLoader.style.display = "inline";

    try {
        // Patient data
        const patient = {
            age: parseFloat(document.getElementById("age").value),
            sex_male: parseInt(document.getElementById("sex_male").value),
            has_htn: document.getElementById("has_htn").checked ? 1 : 0,
            has_dm: document.getElementById("has_dm").checked ? 1 : 0,
            has_ckd: document.getElementById("has_ckd").checked ? 1 : 0,
            serum_sodium: parseFloat(document.getElementById("serum_sodium").value),
            serum_potassium: parseFloat(document.getElementById("serum_potassium").value),
            creatinine: parseFloat(document.getElementById("creatinine").value),
            egfr: parseFloat(document.getElementById("egfr").value),
            hba1c: parseFloat(document.getElementById("hba1c").value),
            fbs: parseFloat(document.getElementById("fbs").value),
            sbp: parseFloat(document.getElementById("sbp").value),
            dbp: parseFloat(document.getElementById("dbp").value),
            bmi: parseFloat(document.getElementById("bmi").value),
        };

        const equipment = document.getElementById("recipe-equipment").value;
        const cuisine = document.getElementById("recipe-cuisine").value;
        const time_limit = document.getElementById("recipe-time").value;

        const res = await fetch(`${API_BASE}/api/generate-recipe`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                patient,
                ingredients: selectedIngredients,
                equipment,
                cuisine,
                time_limit
            })
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || "Failed to generate recipe");
        }

        const data = await res.json();

        contentDiv.innerHTML = markdownToHtml(data.recipe);
        outputSection.style.display = "";
        outputSection.scrollIntoView({ behavior: "smooth", block: "start" });

    } catch (err) {
        showError(err.message);
    } finally {
        btn.disabled = false;
        btnText.style.display = "inline";
        btnLoader.style.display = "none";
    }
}

// ── Markdown Parser ──
function markdownToHtml(md) {
    if (!md) return "";
    let html = escapeHtml(md);

    // Headings
    html = html.replace(/^#### (.+)$/gm, "<h4>$1</h4>");
    html = html.replace(/^### (.+)$/gm, "<h3>$1</h3>");
    html = html.replace(/^## (.+)$/gm, "<h2>$1</h2>");
    html = html.replace(/^# (.+)$/gm, "<h1>$1</h1>");

    // Bold + Italic
    html = html.replace(/\*\*\*(.+?)\*\*\*/g, "<strong><em>$1</em></strong>");
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");

    // Lists
    html = html.replace(/^[•\-\*] (.+)$/gm, "<li>$1</li>");
    html = html.replace(/^\d+\.\s(.+)$/gm, "<li>$1</li>");
    html = html.replace(/((?:<li>.+<\/li>\n?)+)/g, "<ul>$1</ul>");

    // Paragraphs
    html = html.split("\n").map((line) => {
        line = line.trim();
        if (!line) return "";
        if (/^<[hulo]/.test(line)) return line;
        return `<p>${line}</p>`;
    }).join("\n");

    return html;
}

function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

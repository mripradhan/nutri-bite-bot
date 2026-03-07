<<<<<<< HEAD
## 🚀 Quick Start

```bash
# Install
pip install -r requirements.txt

# Run complete pipeline
python main_integration.py
```

**Output**: All required JSON files generated in `outputs/`
=======
# NutriBiteBot: Pantry-to-Plate Clinical Nutrition System 🥗🤖

NutriBiteBot is an intelligent, integrated clinical nutrition system designed to generate safe, personalized recipes based on patient medical data and available pantry ingredients. It features a hierarchical clinical rules engine that manages conflicting nutritional guidelines (e.g., Renal safety vs HTN) and leverages a fine-tuned TabNet model. It now also supports a purely local Supabase instance for securely storing sensitive medical data and recipes.

---

## ✨ Key Features
- **Hierarchical Clinical Rules Engine**: Solves conflicting medical nutritional guidelines (e.g., CKD Stage 3 vs Hypertension potassium caps).
- **TabNet Clinical Model Integration**: Provides patient risk stratification without hardcoded clinical thresholds.
- **Fridge Scanner / Pantry Inventory**: Takes an uploaded image of your fridge, uses Roboflow Computer Vision to detect ingredients, and maps them to the IFCT nutritional database.
- **Recipe Generation**: Uses the Groq API (Llama 3) to generate recipes strictly adhering to safe ingredient portion limitations.
- **Local Secure Storage**: Uses a local Dockerized Supabase instance to store `patients` and `recipes` entirely locally, ensuring data privacy.

---

## 🚀 Quick Start Walkthrough

### 1. Prerequisites
Before you begin, ensure you have the following installed on your machine:
- **Python 3.8+** (for the backend Flask app and ML models)
- **Node.js & npm** (To run the Supabase CLI executable)
- **Docker Desktop** (Required for the local Supabase container stack)

### 2. Environment Setup
Create a `.env` file in the root directory and add your API keys. 
*(Note: Supabase credentials will be populated in the next step)*

```env
GROQ_API_KEY="your_groq_api_key"
ROBOFLOW_API_KEY="your_roboflow_key"
NUTRITION_API_KEY="your_nutrition_api_key"
```

### 3. Start Local Supabase Database
Ensure Docker Desktop is open and running. Then, spin up the local database using the Supabase CLI.

```bash
# This will download the containers and start the local database
npx --yes supabase start
```
*Note: The local Supabase dashboard will be available at `http://127.0.0.1:54333`. You can view your `patients` and `recipes` tables here.*

Once running, update your `.env` file to include the local connection details (as output by the Supabase CLI):
```env
SUPABASE_URL="http://127.0.0.1:54331"
SUPABASE_ANON_KEY="your_local_anon_key"
SUPABASE_SERVICE_ROLE_KEY="your_local_service_role_key"
```

### 4. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the Application
Start the Flask backend API, which also serves the frontend Single Page Application (SPA).

```bash
python app.py
```

### 6. Use the App
Open your browser and navigate to the application:
👉 **[http://localhost:5000](http://localhost:5000)**

From here you can:
1. Input your clinical data (eGFR, HbA1c, etc.) and run the risk stratification models.
2. Upload a photo of your fridge to detect ingredients.
3. Automatically calculate safe ingredient portions based on your clinical restrictions.
4. Generate a delicious recipe!

---

## 🛑 Stopping the Database
When you are done testing, it's good practice to stop your local Supabase containers to free up system resources:
```bash
npx --yes supabase stop
```

---

## 🏗️ System Architecture
```
[Frontend UI (HTML/JS)] <--> [Flask API] <--> [TabNet Models (Model-1/Model-2)]
                                   |                       |
                                   v                       v
                         [Local Supabase DB]      [Clinical Rules Engine]
```
- **Backend Framework**: Flask
- **Machine Learning**: PyTorch TabNet (`pytorch_tabnet`)
- **Database**: Local Supabase (PostgreSQL)
- **Computer Vision**: Roboflow API
- **LLM Engine**: Groq (Llama-3.3-70b-versatile)
>>>>>>> fulltabnetver

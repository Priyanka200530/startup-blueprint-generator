# Startup Blueprint Generator

An AI-powered web application that transforms startup ideas into comprehensive business blueprints using **IBM watsonx Orchestrate**, **IBM Granite Foundation Models**, and **Retrieval-Augmented Generation (RAG)**.

---

## 🚀 Features

- **AI Blueprint Generation** – 19+ section blueprints powered by IBM Granite LLM
- **RAG Knowledge Base** – Watson Discovery retrieves grounded knowledge before generation
- **IBM watsonx Orchestrate** – Agent coordination for multi-step blueprint creation
- **Startup Readiness Dashboard** – Score, risk level, budget summary, and funding recommendation
- **SWOT Analysis** – Strengths, Weaknesses, Opportunities, and Threats
- **Budget Breakdown Chart** – Interactive Chart.js doughnut chart
- **90-Day Roadmap** – Phased execution plan
- **AI Chat Assistant** – Follow-up Q&A powered by IBM watsonx
- **PDF Export** – Download the full blueprint as a PDF
- **Bootstrap 5 Responsive UI** – Professional blue & white startup theme

---

## 🗂️ Project Structure

```
startup-blueprint-generator/
├── app.py                  # Flask application (routes, API calls)
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── README.md               # This file
├── templates/
│   ├── base.html           # Shared layout (navbar, footer)
│   ├── index.html          # Home / landing page
│   ├── generator.html      # Blueprint Generator page
│   ├── 404.html            # 404 error page
│   └── 500.html            # 500 error page
└── static/
    ├── css/
    │   └── style.css       # Custom CSS (Bootstrap + theme)
    ├── js/
    │   ├── main.js         # Global JS (navbar, animations)
    │   └── blueprint.js    # Generator page JS (form, API, charts)
    └── images/             # Image assets directory
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/startup-blueprint-generator.git
cd startup-blueprint-generator
```

### 2. Create and Activate a Virtual Environment

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
```

Open `.env` and fill in your IBM Cloud credentials:

```env
WATSONX_API_KEY=your_ibm_cloud_api_key
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_PROJECT_ID=your_project_id
ORCHESTRATE_AGENT_URL=https://api.ibm.com/watsonx-orchestrate/run/v1/agents/<agent_id>/chat/completions
GRANITE_MODEL_ID=ibm/granite-13b-instruct-v2
DISCOVERY_API_KEY=your_discovery_api_key
DISCOVERY_URL=https://api.us-south.discovery.watson.cloud.ibm.com
DISCOVERY_PROJECT_ID=your_discovery_project_id
DISCOVERY_COLLECTION_ID=your_collection_id
SECRET_KEY=change-me-to-a-random-secret
```

### 5. Run the Application

```bash
python app.py
```

Open your browser at **http://localhost:5000**

---

## 🔑 IBM Cloud Configuration

### IBM watsonx Orchestrate

1. Go to [IBM Cloud](https://cloud.ibm.com) → **watsonx Orchestrate**
2. Create an **Agent** and connect the IBM Granite model
3. Add a **RAG skill** pointing to your Watson Discovery collection
4. Copy the agent's REST endpoint URL → set as `ORCHESTRATE_AGENT_URL`

### IBM Granite (watsonx.ai)

1. Create a **watsonx.ai** project in IBM Cloud
2. Copy the **Project ID** → set as `WATSONX_PROJECT_ID`
3. Choose a Granite model ID (e.g. `ibm/granite-13b-instruct-v2`) → `GRANITE_MODEL_ID`
4. Generate an **IBM Cloud API Key** → set as `WATSONX_API_KEY`

### Watson Discovery (RAG)

1. Create a **Watson Discovery** instance
2. Create a **Project** and upload your startup/business knowledge documents
3. Copy the **Project ID** and **Collection ID** → set in `.env`
4. Generate a **Discovery API Key** → `DISCOVERY_API_KEY`

> **Fallback behaviour:** If `ORCHESTRATE_AGENT_URL` is not set, the application automatically falls back to the watsonx.ai Granite direct API. If neither is configured, an informative error is shown.

---

## 🌐 API Endpoints

| Method | Endpoint                    | Description                         |
|--------|-----------------------------|-------------------------------------|
| GET    | `/`                         | Home / landing page                 |
| GET    | `/generator`                | Blueprint Generator page            |
| POST   | `/api/generate-blueprint`   | Generate a startup blueprint (JSON) |
| POST   | `/api/chat`                 | AI Chat assistant messages          |
| GET    | `/api/health`               | Service health check                |

### `POST /api/generate-blueprint`

**Request body (JSON):**
```json
{
  "startup_name": "EduAI",
  "startup_idea": "An AI-powered adaptive learning platform for K-12 students",
  "industry": "EdTech",
  "target_customers": "Students and parents",
  "country": "India",
  "estimated_budget": "$50,000",
  "startup_stage": "Idea"
}
```

**Response:**
```json
{
  "success": true,
  "blueprint": {
    "executive_summary": "...",
    "swot": { "strengths": [...], "weaknesses": [...], "opportunities": [...], "threats": [...] },
    "readiness_score": 74,
    "...": "..."
  }
}
```

---

## 🚀 Deployment

### Gunicorn (Production)

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

```bash
docker build -t startup-blueprint-generator .
docker run -p 5000:5000 --env-file .env startup-blueprint-generator
```

### IBM Code Engine / Cloud Foundry

```bash
# IBM Cloud Code Engine
ibmcloud ce app create \
  --name startup-blueprint \
  --image us.icr.io/your-namespace/startup-blueprint:latest \
  --port 5000 \
  --env-from-secret startup-secrets
```

---

## 🔒 Security Best Practices

- All IBM credentials are loaded from environment variables — **never hardcoded**
- The `.env` file is in `.gitignore` — never committed
- Input validation on all API endpoints
- HTML-escaped output prevents XSS in blueprint rendering
- IAM token caching with automatic expiry refresh

---

## 🛠️ Tech Stack

| Layer     | Technology                          |
|-----------|-------------------------------------|
| AI Engine | IBM watsonx Orchestrate             |
| LLM       | IBM Granite Foundation Models       |
| RAG       | IBM Watson Discovery                |
| Backend   | Python 3.11, Flask 3.0              |
| Frontend  | HTML5, CSS3, Bootstrap 5, Chart.js  |
| Icons     | Font Awesome 6                      |
| Fonts     | Inter, Poppins (Google Fonts)       |
| PDF       | html2pdf.js                         |

---

## 📄 License

MIT License – see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---
Checkout my linkedin post:https://www.linkedin.com/posts/priyanka-s-astrophile_ibm-ibmwatsonx-watsonxorchestrate-ugcPost-7481262944715173888-aWCa/?utm_source=share&utm_medium=member_android&rcm=ACoAAEfZBd8B9VoulCkKC15n86u_gPtbLjyY5CI
*Built with ❤️ using IBM watsonx AI*

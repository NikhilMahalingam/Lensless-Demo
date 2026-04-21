# Run Backend
cd backend
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
uvicorn app:app

# Run Frontend
cd frontend
npm install
npm run dev
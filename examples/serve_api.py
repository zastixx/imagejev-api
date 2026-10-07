"""Programmatically launch the ImageJev REST API server."""
import uvicorn
from imagejev.server import app

if __name__ == "__main__":
    print("Starting ImageJev API Server on http://localhost:8000")
    print("API Documentation available at http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)

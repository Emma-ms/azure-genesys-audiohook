import asyncio
import logging
import os

import pytest
from app.storage.in_memory_conversation_store import InMemoryConversationStore
from app.websocket_server import WebsocketServer
from app.handler.session_manager import SessionManager
from dotenv import load_dotenv

dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(dotenv_path)

logging.basicConfig(level=logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)

@pytest.fixture
async def server():
    server = WebsocketServer()
    server.conversations_store = InMemoryConversationStore()
    server.session_manager = SessionManager(logging.getLogger(__name__))
    await server.session_manager.create_connections()
    yield server
    await server.session_manager.close_connections()

@pytest.fixture
async def app(server):
    return server.app.test_client()

@pytest.mark.asyncio
async def test_server_fixture(server):
    assert server.app is not None
    assert hasattr(server.app, "test_client")
    assert os.getenv("AZURE_SPEECH_REGION") is not None

@pytest.mark.asyncio
async def test_health_check(app):
    """Test health check endpoint"""
    response = await app.get("/")

    assert response.status_code == 200
    assert await response.data == b'{"status":"healthy"}\n'


@pytest.mark.asyncio
async def test_health_check_valid_json(app):
    """Test if health check endpoint is valid JSON"""
    response = await app.get("/")

    # Check if response data is valid JSON
    data = await response.get_json()

    assert data is not None
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_invalid_route(app):
    """Test invalid route"""
    response = await app.get("/invalid")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_ws_invalid_api_key(server):
    """Test websocket connection with invalid API key"""

    headers = {
        "X-Api-Key": "invalid_key",
        "Audiohook-Session-Id": "test_session",
        "Audiohook-Correlation-Id": "test_correlation",
        "Signature-Input": "test_signature_input",
        "Signature": "test_signature",
    }

    async with server.app.websocket("/audiohook/ws", headers=headers) as ws:
        response = await ws.receive_json()

        assert response["type"] == "disconnect"
        assert response["parameters"]["reason"] == "unauthorized"
        assert response["parameters"]["info"] == "Invalid API Key"


@pytest.mark.asyncio
async def test_ws_valid_connection(app):
    """Test valid websocket connection"""
    headers = {
        "X-Api-Key": os.getenv("WEBSOCKET_SERVER_API_KEY"),
        "Audiohook-Session-Id": "e160e428-53e2-487c-977d-96989bf5c99d",
        "Audiohook-Correlation-Id": "test_correlation",
        "Signature-Input": "test_signature_input",
        "Signature": "test_signature",
    }
    async with app.websocket("/audiohook/ws", headers=headers) as ws:
        # Open Transaction
        # https://developer.genesys.cloud/devapps/audiohook/session-walkthrough#open-transaction
        await ws.send_json(
            {
                "version": "2",
                "type": "open",
                "seq": 1,
                "serverseq": 0,
                "id": "e160e428-53e2-487c-977d-96989bf5c99d",
                "position": "PT0S",
                "parameters": {
                    "organizationId": "d7934305-0972-4844-938e-9060eef73d05",
                    "conversationId": "090eaa2f-72fa-480a-83e0-8667ff89c0ec",
                    "participant": {
                        "id": "883efee8-3d6c-4537-b500-6d7ca4b92fa0",
                        "ani": "+1-555-555-1234",
                        "aniName": "John Doe",
                        "dnis": "+1-800-555-6789",
                    },
                    "media": [
                        {
                            "type": "audio",
                            "format": "PCMU",
                            "channels": ["external", "internal"],
                            "rate": 8000,
                        },
                        {
                            "type": "audio",
                            "format": "PCMU",
                            "channels": ["external"],
                            "rate": 8000,
                        },
                        {
                            "type": "audio",
                            "format": "PCMU",
                            "channels": ["internal"],
                            "rate": 8000,
                        },
                    ],
                    "language": "en-US",
                },
            }
        )

        response = await ws.receive_json()

        assert response["type"] == "opened"

@pytest.mark.asyncio
async def test_ws_audio_processing(app):
    """Test valid websocket connection"""
    API_KEY= os.getenv("WEBSOCKET_SERVER_API_KEY")
    CONVERSATION_ID = "090eaa2f-72fa-480a-83e0-8667ff89c0ec"
    headers = {
        "X-Api-Key": API_KEY,
        "Audiohook-Session-Id": "e160e428-53e2-487c-977d-96989bf5c99d",
        "Audiohook-Correlation-Id": "test_correlation",
        "Signature-Input": "test_signature_input",
        "Signature": "test_signature",
    }
    
    async with app.websocket("/audiohook/ws", headers=headers) as ws:
        await ws.send_json(
            {
                "version": "2",
                "type": "open",
                "seq": 1,
                "serverseq": 0,
                "id": "e160e428-53e2-487c-977d-96989bf5c99d",
                "position": "PT0S",
                "parameters": {
                    "organizationId": "d7934305-0972-4844-938e-9060eef73d05",
                    "conversationId": CONVERSATION_ID,
                    "participant": {
                        "id": "883efee8-3d6c-4537-b500-6d7ca4b92fa0",
                        "ani": "+1-555-555-1234",
                        "aniName": "John Doe",
                        "dnis": "+1-800-555-6789",
                    },
                    "media": [
                        {
                            "type": "audio",
                            "format": "PCMU",
                            "channels": ["external", "internal"],
                            "rate": 8000,
                        }
                    ],
                    "language": "en-US",
                },
            }
        )

        response = await ws.receive_json()
        logging.info("WebSocket response: %s", response)

        assert response["type"] == "opened"

        # Read and send WAV file in chunks
        file_path = os.path.join(os.path.dirname(__file__), "test.wav")
        with open(file_path, "rb") as f:
            while chunk := f.read(1024):
                await ws.send(chunk)
                await asyncio.sleep(0.01) 
        try:
            while True:
                response = await asyncio.wait_for(ws.receive_json(), timeout=20)
                logging.info("WebSocket response: %s", response)
                if response.get("type") == "stop":
                    break
        except asyncio.TimeoutError:
            logging.warning("No response from websocket (timeout).")

        # response = await app.get(f"/api/conversations?key={API_KEY}")

        # assert response.status_code == 200
        # conversations = await response.get_json()
        # logging.info("Conversations:", conversations)

        # optional, closed the connection
# src/ai_trading/websocket_streaming_service.py
"""
WebSocket-based real-time data streaming service
"""

import asyncio
import json
from datetime import datetime
from typing import Set, Dict, Any


class WebSocketStreamingService:
    """WebSocket service for real-time data streaming"""

    def __init__(self):
        self.connected_clients: Set = set()
        self.is_running = False
        self.stream_data = {}

    async def start_server(self, host: str = "localhost", port: int = 8765):
        """Start the WebSocket server"""
        self.is_running = True
        print(f"WebSocket server starting on {host}:{port}")

        # Simulate server startup
        await asyncio.sleep(0.1)
        print("WebSocket server started successfully")

    async def stop_server(self):
        """Stop the WebSocket server"""
        self.is_running = False
        self.connected_clients.clear()
        print("WebSocket server stopped")

    async def broadcast_data(self, data: Dict[str, Any]):
        """Broadcast data to all connected clients"""
        if not self.connected_clients:
            return

        message = json.dumps({
            'type': 'market_data',
            'data': data,
            'timestamp': datetime.now().isoformat()
        })

        print(f"Broadcasting to {len(self.connected_clients)} clients")

        # Simulate broadcasting
        for client in list(self.connected_clients):
            try:
                # Simulate sending data
                self.stream_data[f"client_{id(client)}"] = data
            except Exception as e:
                print(f"Error broadcasting to client: {e}")
                self.connected_clients.discard(client)

    async def handle_client_connection(self, client_id: str):
        """Handle new client connection"""
        self.connected_clients.add(client_id)
        print(f"Client {client_id} connected. Total clients: {len(self.connected_clients)}")

        # Send welcome message
        welcome_data = {
            'type': 'welcome',
            'message': 'Connected to MarketPulse streaming service',
            'client_id': client_id
        }
        await self.broadcast_data(welcome_data)

    async def handle_client_disconnect(self, client_id: str):
        """Handle client disconnection"""
        self.connected_clients.discard(client_id)
        print(f"Client {client_id} disconnected. Total clients: {len(self.connected_clients)}")

    def get_service_stats(self) -> Dict:
        """Get service statistics"""
        return {
            'is_running': self.is_running,
            'connected_clients': len(self.connected_clients),
            'total_broadcasts': len(self.stream_data),
            'last_activity': datetime.now()
        }
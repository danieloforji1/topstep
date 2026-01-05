"""
Operations API
HTTP endpoints for status, metrics, manual halt/flatten, parameter changes
"""
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import os

logger = logging.getLogger(__name__)

app = FastAPI(title="Grid Strategy API", version="1.0.0")

# Enable CORS for dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (dashboard UI)
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    
    @app.get("/")
    async def serve_dashboard():
        """Serve the dashboard UI"""
        return FileResponse(os.path.join(static_dir, "index.html"))
    
    @app.get("/favicon.ico")
    async def favicon():
        """Return 204 No Content for favicon (prevents 404 errors)"""
        from fastapi.responses import Response
        return Response(status_code=204)


class StatusResponse(BaseModel):
    """Status response model"""
    status: str
    timestamp: datetime
    trading_enabled: bool
    dry_run: bool
    daily_pnl: float
    total_pnl: float
    net_exposure: float
    open_orders: int
    positions: Dict[str, Any]


class FlattenRequest(BaseModel):
    """Flatten request model"""
    reason: Optional[str] = None


class PauseRequest(BaseModel):
    """Pause request model"""
    pause: bool


# Global strategy instance (set by main)
strategy_instance = None


def set_strategy_instance(strategy):
    """Set the strategy instance for API access"""
    global strategy_instance
    strategy_instance = strategy


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Grid Strategy API", "version": "1.0.0"}


@app.get("/ops/status")
async def get_status():
    """Get current strategy status"""
    if strategy_instance is None:
        # Return a default status when strategy isn't initialized yet
        return {
            "status": "initializing",
            "timestamp": datetime.now().isoformat(),
            "trading_enabled": False,
            "dry_run": True,
            "account_balance": 0.0,
            "daily_pnl": 0.0,
            "total_pnl": 0.0,
            "net_exposure": 0.0,
            "open_orders": 0,
            "open_orders_list": [],
            "drawdown": 0.0,
            "current_price": None,
            "primary_symbol": None,
            "positions": {},
            "message": "Strategy is initializing. Please wait..."
        }
    
    try:
        status = strategy_instance.get_status()
        # Convert to dict if it's already a dict, or use response model
        if isinstance(status, dict):
            return status
        return StatusResponse(**status).dict()
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ops/flatten")
async def flatten_positions(request: FlattenRequest):
    """Force flatten all positions"""
    if strategy_instance is None:
        raise HTTPException(status_code=503, detail="Strategy not initialized")
    
    try:
        reason = request.reason or "Manual flatten via API"
        strategy_instance.emergency_flatten(reason)
        return {"success": True, "message": f"Flattened positions: {reason}"}
    except Exception as e:
        logger.error(f"Error flattening: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ops/pause")
async def pause_strategy(request: PauseRequest):
    """Pause or resume strategy"""
    if strategy_instance is None:
        raise HTTPException(status_code=503, detail="Strategy not initialized")
    
    try:
        strategy_instance.set_paused(request.pause)
        return {"success": True, "paused": request.pause}
    except Exception as e:
        logger.error(f"Error pausing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ops/metrics")
async def get_metrics():
    """Get strategy metrics"""
    if strategy_instance is None:
        raise HTTPException(status_code=503, detail="Strategy not initialized")
    
    try:
        metrics = strategy_instance.get_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}


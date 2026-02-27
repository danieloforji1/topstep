"""
Decision Engine - Core AI agent logic
"""
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import asyncpg
import redis.asyncio as redis

from .llm_client import LLMClient
from .risk_gate import RiskGate
from .strategy_analyzer import StrategyAnalyzer
from .topstep_client import TopstepClient

logger = logging.getLogger(__name__)


class DecisionEngine:
    """AI Decision Engine - Makes trading decisions using LLM"""
    
    def __init__(
        self,
        llm_client: LLMClient,
        risk_gate: RiskGate,
        strategy_analyzer: StrategyAnalyzer,
        topstep_client: TopstepClient,
        db_pool: asyncpg.Pool,
        redis_client: redis.Redis
    ):
        self.llm_client = llm_client
        self.risk_gate = risk_gate
        self.strategy_analyzer = strategy_analyzer
        self.topstep_client = topstep_client
        self.db_pool = db_pool
        self.redis_client = redis_client
    
    async def make_decision(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        current_positions: List[Dict[str, Any]],
        account_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Main decision-making function
        
        Returns:
            {
                "decision": "strategy_select" | "entry" | "exit" | "hold",
                "strategy_name": str,
                "action": {...},
                "reasoning": str,
                "confidence": float,
                "risk_check_passed": bool,
                "timestamp": datetime,
                "llm_response": {...}
            }
        """
        # 1. Gather context
        context = await self._gather_context(symbol, market_data, current_positions, account_state)
        
        # 2. Build prompt for LLM
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(context)
        
        # 3. Get decision from LLM
        try:
            llm_response = await self.llm_client.generate_decision(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3  # Low temperature for consistent decisions
            )
        except Exception as e:
            logger.error(f"LLM error: {e}")
            # Fallback to conservative decision
            return {
                "decision": "hold",
                "strategy_name": None,
                "action": None,
                "reasoning": f"LLM error: {e}. Holding position for safety.",
                "confidence": 0.0,
                "risk_check_passed": True,
                "timestamp": datetime.utcnow(),
                "llm_response": None
            }
        
        # 4. Validate and structure decision
        decision = self._parse_llm_response(llm_response, context)
        
        # 5. Risk check
        risk_check = await self.risk_gate.check_decision(decision, account_state)
        decision["risk_check_passed"] = risk_check["passed"]
        
        if not risk_check["passed"]:
            logger.warning(f"Risk check failed: {risk_check['reason']}")
            decision["decision"] = "hold"
            decision["action"] = None
            decision["reasoning"] = f"Risk check failed: {risk_check['reason']}. {decision['reasoning']}"
        
        decision["llm_response"] = llm_response
        decision["timestamp"] = datetime.utcnow()
        
        return decision
    
    async def _gather_context(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        current_positions: List[Dict[str, Any]],
        account_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Gather all context for decision-making"""
        # Get strategy performance
        strategies = await self.strategy_analyzer.get_strategies()
        strategy_performance = {}
        for strategy in strategies:
            perf = await self.strategy_analyzer.get_strategy_performance(strategy["name"])
            strategy_performance[strategy["name"]] = perf
        
        # Get recent decisions
        recent_decisions = await self._get_recent_decisions(limit=10)
        
        # Get risk status
        risk_status = await self.risk_gate.get_status()
        
        return {
            "symbol": symbol,
            "market_data": market_data,
            "current_positions": current_positions,
            "account_state": account_state,
            "strategies": strategies,
            "strategy_performance": strategy_performance,
            "recent_decisions": recent_decisions,
            "risk_status": risk_status
        }
    
    def _build_system_prompt(self) -> str:
        """Build system prompt that defines the agent's role"""
        return """You are an expert quantitative trading analyst for TopstepX futures trading.

Your role:
1. Analyze market conditions and select the best strategy
2. Make entry/exit decisions based on strategy rules
3. Always prioritize risk management and Topstep rules compliance

Available Strategies:
1. grid_strategy: Volatility-adaptive grid trading with cross-asset hedging (MES/MNQ)
   - Best for: Mean-reverting markets, low volatility
   - Entry: Grid levels based on ATR
   - Exit: Quick reversion or time stop
   
2. multi_timeframe: Multi-timeframe convergence (1m, 5m, 15m)
   - Best for: Trending markets with clear direction
   - Entry: When all timeframes agree (weighted signal > 0.2)
   - Exit: Divergence or stop/target
   
3. liquidity_provision: Smart market making
   - Best for: Balanced markets, capturing spread
   - Entry: When favorable fill probability > 60%
   - Exit: Quick profit or adverse selection detected
   
4. optimal_stopping: Optimal entry/exit timing
   - Best for: High-quality setups only
   - Entry: After seeing 3+ opportunities, take best one
   - Exit: Optimal stopping condition met
   
5. statarb: Statistical arbitrage (GC/MGC)
   - Best for: Pairs trading, mean reversion
   - Entry: Spread z-score > 2.0
   - Exit: Spread returns to mean

Topstep Rules (CRITICAL):
- Maximum Loss Limit (MLL): $2,000 trailing from account high
- Daily Loss Limit (DLL): $2,000 per day
- No overnight positions
- Must close all positions before market close

Your responses must be valid JSON with this structure:
{
    "decision": "strategy_select" | "entry" | "exit" | "hold",
    "strategy_name": "grid_strategy" | "multi_timeframe" | ... | null,
    "action": {
        "type": "entry" | "exit" | null,
        "side": "BUY" | "SELL" | null,
        "quantity": integer,
        "price": float | null,
        "stop_loss": float | null,
        "take_profit": float | null
    },
    "reasoning": "Detailed explanation of your decision",
    "confidence": 0.0-1.0
}"""
    
    def _build_user_prompt(self, context: Dict[str, Any]) -> str:
        """Build user prompt with current context"""
        prompt = f"""Current Market Context:

Symbol: {context['symbol']}
Price: {context['market_data'].get('price', 'N/A')}
ATR: {context['market_data'].get('atr', 'N/A')}
Volatility: {context['market_data'].get('volatility', 'N/A')}
Volume: {context['market_data'].get('volume', 'N/A')}

Account State:
Balance: ${context['account_state'].get('balance', 0):.2f}
Daily PnL: ${context['account_state'].get('daily_pnl', 0):.2f}
Drawdown: ${context['account_state'].get('drawdown', 0):.2f}
Equity High: ${context['account_state'].get('equity_high', 0):.2f}

Current Positions: {len(context['current_positions'])}
"""
        if context['current_positions']:
            for pos in context['current_positions']:
                prompt += f"  - {pos.get('strategy')}: {pos.get('side')} {pos.get('quantity')} @ {pos.get('entry_price')}\n"
        else:
            prompt += "  - No open positions\n"
        
        prompt += "\nStrategy Performance:\n"
        for name, perf in context['strategy_performance'].items():
            prompt += f"  - {name}: Win Rate {perf.get('win_rate', 0):.1%}, Sharpe {perf.get('sharpe', 0):.2f}\n"
        
        prompt += "\nRisk Status:\n"
        risk = context['risk_status']
        prompt += f"  - MLL Distance: ${risk.get('mll_distance', 0):.2f}\n"
        prompt += f"  - DLL Remaining: ${risk.get('dll_remaining', 0):.2f}\n"
        prompt += f"  - Can Trade: {risk.get('can_trade', False)}\n"
        
        prompt += "\nWhat should we do? Provide your decision as JSON."
        
        return prompt
    
    def _parse_llm_response(self, llm_response: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and validate LLM response"""
        # Extract decision components
        decision = {
            "decision": llm_response.get("decision", "hold"),
            "strategy_name": llm_response.get("strategy_name"),
            "action": llm_response.get("action"),
            "reasoning": llm_response.get("reasoning", "No reasoning provided"),
            "confidence": float(llm_response.get("confidence", 0.0))
        }
        
        # Validate decision type
        valid_decisions = ["strategy_select", "entry", "exit", "hold"]
        if decision["decision"] not in valid_decisions:
            logger.warning(f"Invalid decision type: {decision['decision']}, defaulting to 'hold'")
            decision["decision"] = "hold"
        
        # Validate confidence
        decision["confidence"] = max(0.0, min(1.0, decision["confidence"]))
        
        return decision
    
    async def _get_recent_decisions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent decisions from database"""
        if not self.db_pool:
            return []
        
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT decision_type, strategy_name, symbol, reasoning, confidence, timestamp
                    FROM agent_decisions
                    ORDER BY timestamp DESC
                    LIMIT $1
                """, limit)
                
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error fetching recent decisions: {e}")
            return []


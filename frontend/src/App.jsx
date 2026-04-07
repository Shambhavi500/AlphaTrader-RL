import React, { useState, useEffect, useMemo } from 'react';
import axios from 'axios';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area
} from 'recharts';
import {
    TrendingUp, TrendingDown, Clock, Wallet, BarChart3, MessageSquare,
    Play, RotateCcw, ShieldCheck, AlertCircle, ChevronRight, Activity
} from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

/** Utility for merging tailwind classes */
function cn(...inputs) {
    return twMerge(clsx(inputs));
}

// ──────────────────────────────────────────────────────────────────────────────
// Mock Data for Design (in case API is unreachable during initial load)
// ──────────────────────────────────────────────────────────────────────────────
const MOCK_TRADES = [
    { action: 'BUY', price: 104.2, timestamp: '10:15 AM', return: '+0.5%' },
    { action: 'SELL', price: 108.5, timestamp: '11:30 AM', return: '+4.1%' },
    { action: 'BUY', price: 106.8, timestamp: '01:45 PM', return: '-0.2%' },
];

export default function App() {
    const [symbol, setSymbol] = useState('TATASTEEL.NS');
    const [loading, setLoading] = useState(false);
    const [status, setStatus] = useState(null);
    const [explanation, setExplanation] = useState('Initialize an agent to see AI-driven trade rationale.');
    const [trades, setTrades] = useState([]);
    const [equityHistory, setEquityHistory] = useState([]);
    const [isLive, setIsLive] = useState(false);

    // ──────────────────────────────────────────────────────────────────────────────
    // API Interactions
    // ──────────────────────────────────────────────────────────────────────────────

    const fetchStatus = async () => {
        try {
            const resp = await axios.get(`/api/portfolio/${symbol}`);
            setStatus(resp.data);
        } catch (err) {
            console.warn("Portfolio status fetch failed - likely uninitialized");
        }
    };

    const stepTrade = async () => {
        setLoading(true);
        try {
            // For the demo/paper trader, we'd normally get live data from a feed.
            // Here we simulate the request to the paper trader.
            // In a real environment, we'd fetch the current state from the market.
            const mockObs = Array(36).fill(0).map(() => Math.random() * 2 - 1);
            const mockPrice = 100 + Math.random() * 20;

            const resp = await axios.post('/api/trade', {
                symbol,
                features: { "rsi_14": 45 + Math.random() * 10, "volume_ratio": 1.2 },
                current_price: mockPrice,
                observation: [...mockObs, 0, 0, 0, 0, 1] // Adding dummy portfolio state for 41 dims
            });

            const result = resp.data;
            setExplanation(result.explanation);

            if (result.trade_occurred) {
                setTrades(prev => [result, ...prev].slice(0, 10));
            }

            setEquityHistory(prev => [...prev, {
                time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
                value: result.portfolio_value
            }]);

            setStatus({
                symbol: result.symbol,
                portfolio_value: result.portfolio_value,
                cash: result.cash,
                shares_held: result.shares_held,
                total_trades: trades.length + (result.trade_occurred ? 1 : 0)
            });

        } catch (err) {
            console.error("Trade step failed", err);
            setExplanation("Error communicating with AI backend. Ensure FastAPI is running.");
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchStatus();
        // Start with initial history point
        setEquityHistory([{ time: 'Start', value: 100000 }]);
    }, [symbol]);

    // ──────────────────────────────────────────────────────────────────────────────
    // UI Components
    // ──────────────────────────────────────────────────────────────────────────────

    const StatCard = ({ title, value, icon: Icon, colorClass, trend }) => (
        <div className="glass-card flex items-center justify-between">
            <div>
                <p className="text-slate-400 text-sm font-medium">{title}</p>
                <h3 className="text-2xl font-bold mt-1 text-white">{value}</h3>
                {trend && (
                    <p className={cn("text-xs font-semibold mt-1", trend > 0 ? "text-emerald-400" : "text-rose-400")}>
                        {trend > 0 ? '↑' : '↓'} {Math.abs(trend).toFixed(2)}%
                    </p>
                )}
            </div>
            <div className={cn("p-3 rounded-lg bg-slate-800", colorClass)}>
                <Icon className="w-6 h-6" />
            </div>
        </div>
    );

    return (
        <div className="min-h-screen p-4 md:p-8 flex flex-col gap-8 max-w-7xl mx-auto">
            {/* Header */}
            <header className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                    <h1 className="text-3xl font-black tracking-tight text-white flex items-center gap-2">
                        AlphaTrader<span className="text-indigo-500">RL</span>
                        <span className="bg-indigo-500/10 text-indigo-400 text-[10px] px-2 py-0.5 rounded-full border border-indigo-500/20 uppercase tracking-widest font-bold">Live Portal</span>
                    </h1>
                    <p className="text-slate-400 text-sm mt-1">Autonomous PPO-Agent with LLM Decisioning</p>
                </div>

                <div className="flex items-center gap-3">
                    <div className="flex bg-slate-900 border border-slate-800 rounded-lg p-1">
                        <input
                            value={symbol}
                            onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                            className="bg-transparent border-none focus:ring-0 text-sm font-bold text-white px-3 w-32"
                        />
                    </div>
                    <button
                        onClick={stepTrade}
                        disabled={loading}
                        className="btn-primary flex items-center gap-2"
                    >
                        {loading ? <Activity className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                        Step Agent
                    </button>
                </div>
            </header>

            {/* Stats Grid */}
            <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <StatCard
                    title="Portfolio Value"
                    value={status ? `₹${status.portfolio_value.toLocaleString()}` : '₹100,000'}
                    icon={Wallet}
                    colorClass="text-indigo-400"
                    trend={status ? ((status.portfolio_value - 100000) / 1000) : 0}
                />
                <StatCard
                    title="Total Profit/Loss"
                    value={status ? `${((status.portfolio_value - 100000) / 1000).toFixed(2)}%` : '0.00%'}
                    icon={TrendingUp}
                    colorClass="text-emerald-400"
                />
                <StatCard
                    title="Active Ticker"
                    value={symbol}
                    icon={Activity}
                    colorClass="text-amber-400"
                />
                <StatCard
                    title="Total Trades"
                    value={status ? status.total_trades : '0'}
                    icon={BarChart3}
                    colorClass="text-sky-400"
                />
            </section>

            {/* Main Content Layout */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">

                {/* Chart Column */}
                <div className="lg:col-span-2 flex flex-col gap-8">
                    <div className="glass-card h-[400px]">
                        <div className="flex items-center justify-between mb-6">
                            <h2 className="text-lg font-bold text-white flex items-center gap-2">
                                <TrendingUp className="w-5 h-5 text-indigo-400" />
                                Equity Curve
                            </h2>
                            <div className="flex items-center gap-2 text-xs text-slate-400">
                                <span className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-indigo-500"></div> Portfolio</span>
                            </div>
                        </div>
                        <div className="h-[280px] w-full mt-4">
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={equityHistory}>
                                    <defs>
                                        <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                                            <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <XAxis dataKey="time" stroke="#475569" fontSize={10} axisLine={false} tickLine={false} />
                                    <YAxis domain={['auto', 'auto']} stroke="#475569" fontSize={10} axisLine={false} tickLine={false} tickFormatter={(val) => `₹${val / 1000}k`} />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '8px' }}
                                        itemStyle={{ color: '#fff' }}
                                    />
                                    <Area type="monotone" dataKey="value" stroke="#6366f1" strokeWidth={2} fillOpacity={1} fill="url(#colorValue)" animationDuration={1000} />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* Trade Table */}
                    <div className="glass-card overflow-hidden">
                        <h2 className="text-lg font-bold text-white flex items-center gap-2 mb-4">
                            <Clock className="w-5 h-5 text-sky-400" />
                            Recent Trade History
                        </h2>
                        <div className="overflow-x-auto">
                            <table className="w-full text-left text-sm">
                                <thead className="text-slate-500 uppercase text-[10px] tracking-widest font-bold">
                                    <tr>
                                        <th className="pb-3 pr-4">Action</th>
                                        <th className="pb-3 pr-4">Execution Price</th>
                                        <th className="pb-3 pr-4">Shares</th>
                                        <th className="pb-3">Status</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-slate-800">
                                    {trades.length > 0 ? trades.map((trade, i) => (
                                        <tr key={i} className="group hover:bg-slate-800/30 transition-colors">
                                            <td className="py-4">
                                                <span className={cn(
                                                    "px-2 py-1 rounded text-[10px] font-bold uppercase",
                                                    trade.action === 'BUY' ? "bg-emerald-500/10 text-emerald-400" : "bg-rose-500/10 text-rose-400"
                                                )}>
                                                    {trade.action}
                                                </span>
                                            </td>
                                            <td className="py-4 font-mono text-slate-300">₹{trade.price.toLocaleString()}</td>
                                            <td className="py-4 text-slate-300">{trade.shares_held}</td>
                                            <td className="py-4">
                                                <span className="flex items-center gap-1.5 text-emerald-400">
                                                    <ShieldCheck className="w-4 h-4" />
                                                    <span className="text-xs">Executed</span>
                                                </span>
                                            </td>
                                        </tr>
                                    )) : (
                                        <tr>
                                            <td colSpan="4" className="py-8 text-center text-slate-500 italic">No trades recorded for this session.</td>
                                        </tr>
                                    )}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>

                {/* Sidebar Column */}
                <div className="flex flex-col gap-8">
                    {/* AI Explainer */}
                    <div className="glass-card min-h-[250px] border-indigo-500/30 ring-1 ring-indigo-500/20">
                        <div className="flex items-center gap-2 mb-4">
                            <div className="p-1.5 bg-indigo-500/20 rounded-lg">
                                <MessageSquare className="w-5 h-5 text-indigo-400" />
                            </div>
                            <h2 className="text-lg font-bold text-white tracking-tight">AI Explainer</h2>
                            <div className="ml-auto w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]"></div>
                        </div>
                        <div className="relative">
                            <div className="absolute -left-6 top-1/2 -translate-y-1/2 w-1 h-32 bg-indigo-500/20 rounded-full blur-sm"></div>
                            <p className="text-slate-300 leading-relaxed text-sm italic py-2 pl-2 border-l border-slate-700">
                                "{explanation}"
                            </p>
                        </div>
                        <div className="mt-6 pt-4 border-t border-slate-800 flex items-center justify-between">
                            <span className="text-[10px] text-slate-500 font-bold uppercase tracking-wider">Powered by xAI / Grok</span>
                            <BarChart3 className="w-4 h-4 text-slate-600" />
                        </div>
                    </div>

                    {/* Manual Controls */}
                    <div className="glass-card">
                        <h2 className="text-lg font-bold text-white flex items-center gap-2 mb-4">
                            <RotateCcw className="w-5 h-5 text-rose-400" />
                            Manual Override
                        </h2>
                        <div className="grid grid-cols-1 gap-3">
                            <button className="flex items-center justify-between p-3 rounded-lg bg-slate-800 hover:bg-emerald-500/10 border border-slate-700 hover:border-emerald-500/50 transition-all group">
                                <span className="text-sm font-semibold group-hover:text-emerald-400">Manual BUY</span>
                                <ChevronRight className="w-4 h-4 text-slate-500 group-hover:text-emerald-400" />
                            </button>
                            <button className="flex items-center justify-between p-3 rounded-lg bg-slate-800 hover:bg-rose-500/10 border border-slate-700 hover:border-rose-500/50 transition-all group">
                                <span className="text-sm font-semibold group-hover:text-rose-400">Manual SELL</span>
                                <ChevronRight className="w-4 h-4 text-slate-500 group-hover:text-rose-400" />
                            </button>
                            <button className="flex items-center justify-between p-3 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 transition-all group">
                                <span className="text-sm font-semibold text-slate-400">Force HOLD</span>
                                <ChevronRight className="w-4 h-4 text-slate-500" />
                            </button>
                        </div>
                        <div className="mt-4 flex items-start gap-2 p-3 bg-rose-500/5 border border-rose-500/20 rounded-lg">
                            <AlertCircle className="w-4 h-4 text-rose-400 mt-0.5 shrink-0" />
                            <p className="text-[10px] text-rose-300/80 leading-snug">
                                Warning: Manual overrides may degrade the RL agent's performance and break its decision consistency.
                            </p>
                        </div>
                    </div>
                </div>

            </div>
        </div>
    );
}

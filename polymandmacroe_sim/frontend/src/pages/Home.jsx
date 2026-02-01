import React from 'react';
import { ArrowRight, Globe, TrendingUp, AlertTriangle } from 'lucide-react';
import { Link } from 'react-router-dom';

const StatCard = ({ title, value, sub, icon: Icon, color }) => (
    <div className="bg-surface border border-gray-700 rounded-xl p-5 hover:border-gray-600 transition-all">
        <div className="flex justify-between items-start mb-4">
            <div>
                <p className="text-gray-400 text-sm font-medium">{title}</p>
                <h3 className="text-2xl font-bold mt-1">{value}</h3>
            </div>
            <div className={`p-2 rounded-lg bg-opacity-10 ${color.bg} ${color.text}`}>
                <Icon size={20} />
            </div>
        </div>
        <p className="text-xs text-gray-500">{sub}</p>
    </div>
);

const Home = () => {
    return (
        <div className="space-y-8">
            {/* Hero Section */}
            <div>
                <h1 className="text-3xl font-bold mb-2">Welcome back, Trader</h1>
                <p className="text-gray-400">Here's your daily briefing across global macro and prediction markets.</p>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <StatCard
                    title="Global Health"
                    value="Neutral (C+)"
                    sub="3 Countries upgraded this week"
                    icon={Globe}
                    color={{ bg: 'bg-blue-500', text: 'text-blue-500' }}
                />
                <StatCard
                    title="Active Arbitrages"
                    value="12"
                    sub="+3.4% Avg. Edge"
                    icon={TrendingUp}
                    color={{ bg: 'bg-emerald-500', text: 'text-emerald-500' }}
                />
                <StatCard
                    title="Risk Alerts"
                    value="2"
                    sub="High Volatility in JPY pairs"
                    icon={AlertTriangle}
                    color={{ bg: 'bg-amber-500', text: 'text-amber-500' }}
                />
            </div>

            {/* Quick Actions */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Link to="/macro" className="group relative overflow-hidden rounded-2xl bg-gradient-to-br from-blue-900/50 to-slate-900 border border-blue-800/50 p-6 hover:border-blue-500/50 transition-all">
                    <div className="relative z-10">
                        <h3 className="text-xl font-semibold mb-2 flex items-center">
                            Launch Macro Grader
                            <ArrowRight className="ml-2 opacity-0 group-hover:opacity-100 transform translate-x-[-10px] group-hover:translate-x-0 transition-all" size={18} />
                        </h3>
                        <p className="text-gray-400 text-sm">Analyze GDP, CPI, and Unemployment Z-Scores to find fundamental divergences.</p>
                    </div>
                    <div className="absolute right-0 bottom-0 opacity-10 transform translate-x-10 translate-y-10">
                        <Globe size={150} />
                    </div>
                </Link>

                <Link to="/polymarket" className="group relative overflow-hidden rounded-2xl bg-gradient-to-br from-emerald-900/50 to-slate-900 border border-emerald-800/50 p-6 hover:border-emerald-500/50 transition-all">
                    <div className="relative z-10">
                        <h3 className="text-xl font-semibold mb-2 flex items-center">
                            Launch Edge Finder
                            <ArrowRight className="ml-2 opacity-0 group-hover:opacity-100 transform translate-x-[-10px] group-hover:translate-x-0 transition-all" size={18} />
                        </h3>
                        <p className="text-gray-400 text-sm">Scan Polymarket for +EV bets using Kelly Criterion sizing.</p>
                    </div>
                    <div className="absolute right-0 bottom-0 opacity-10 transform translate-x-10 translate-y-10">
                        <TrendingUp size={150} />
                    </div>
                </Link>
            </div>
        </div>
    );
};

export default Home;

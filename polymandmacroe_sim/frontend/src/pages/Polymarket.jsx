import React from 'react';
import { TrendingUp } from 'lucide-react';

const Polymarket = () => {
    return (
        <div>
            <div className="mb-6">
                <h2 className="text-2xl font-bold flex items-center">
                    <TrendingUp className="mr-2 text-emerald-500" />
                    Polymarket Edge Finder
                </h2>
                <p className="text-gray-400">Arbitrage opportunities and +EV bets detected in real-time.</p>
            </div>

            <div className="bg-surface border border-gray-700 rounded-lg overflow-hidden">
                <table className="w-full text-left">
                    <thead className="bg-gray-900/50 text-gray-400 text-sm">
                        <tr>
                            <th className="p-4">Market</th>
                            <th className="p-4">Outcome</th>
                            <th className="p-4">Price</th>
                            <th className="p-4 text-emerald-500">Edge</th>
                            <th className="p-4">Kelly Size</th>
                            <th className="p-4">Action</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-700/50">
                        {/* Placeholder Rows */}
                        <tr className="hover:bg-white/5 transition-colors">
                            <td className="p-4 font-medium">Bitcoin &gt; $100k by 2025</td>
                            <td className="p-4"><span className="text-emerald-400">YES</span></td>
                            <td className="p-4">32c</td>
                            <td className="p-4 text-emerald-500 font-bold">+4.5%</td>
                            <td className="p-4">$150</td>
                            <td className="p-4">
                                <button className="bg-blue-600 hover:bg-blue-500 text-white px-3 py-1 rounded text-xs font-bold">
                                    BET
                                </button>
                            </td>
                        </tr>
                        <tr className="hover:bg-white/5 transition-colors">
                            <td className="p-4 font-medium">Fed Rate Cut in March</td>
                            <td className="p-4"><span className="text-red-400">NO</span></td>
                            <td className="p-4">85c</td>
                            <td className="p-4 text-emerald-500 font-bold">+1.2%</td>
                            <td className="p-4">$450</td>
                            <td className="p-4">
                                <button className="bg-blue-600 hover:bg-blue-500 text-white px-3 py-1 rounded text-xs font-bold">
                                    BET
                                </button>
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <div className="mt-4 text-center text-xs text-gray-500">
                Showing mock data. Connect backend API to populate live trades.
            </div>
        </div>
    );
};

export default Polymarket;

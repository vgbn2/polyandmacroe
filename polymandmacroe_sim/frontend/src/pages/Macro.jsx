import React, { useEffect, useState } from 'react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';
import { AlertCircle } from 'lucide-react';

const Macro = () => {
    const [heatmap, setHeatmap] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetch('/api/macro/heatmap')
            .then(res => res.json())
            .then(data => {
                setHeatmap(data);
                setLoading(false);
            })
            .catch(err => {
                console.error("Failed to fetch heatmap:", err);
                setLoading(false);
            });
    }, []);

    if (loading) return <div className="p-8 text-center text-gray-500">Loading Global Data...</div>;

    return (
        <div className="space-y-6">
            <div className="flex justify-between items-center">
                <div>
                    <h2 className="text-2xl font-bold">Global Macro Heatmap</h2>
                    <p className="text-gray-400">Real-time health grading based on GDP, CPI, and Employment Z-Scores.</p>
                </div>
            </div>

            {/* Scorecard Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {heatmap.map((country) => (
                    <div key={country.code} className="bg-surface border border-gray-700 rounded-lg p-5">
                        <div className="flex justify-between items-start">
                            <div>
                                <h3 className="text-lg font-bold">{country.name}</h3>
                                <span className={`inline-block px-2 py-1 rounded text-xs font-bold mt-1 ${country.grade === 'A' ? 'bg-emerald-500/20 text-emerald-500' :
                                        country.grade === 'B' ? 'bg-blue-500/20 text-blue-500' :
                                            country.grade === 'F' ? 'bg-red-500/20 text-red-500' :
                                                'bg-gray-500/20 text-gray-400'
                                    }`}>
                                    Grade: {country.grade}
                                </span>
                            </div>
                            <div className="text-right">
                                <p className="text-2xl font-mono text-white">{country.score.toFixed(2)}</p>
                                <p className="text-xs text-gray-500">Bias Score</p>
                            </div>
                        </div>

                        <div className="mt-4 pt-4 border-t border-gray-700/50">
                            {/* Mini trendline or details could go here */}
                            <p className="text-sm text-gray-400">Updated: {new Date(country.date).toLocaleDateString()}</p>
                        </div>
                    </div>
                ))}
            </div>

            {heatmap.length === 0 && (
                <div className="p-8 bg-surface/50 border border-dashed border-gray-700 rounded-xl text-center">
                    <AlertCircle className="mx-auto text-gray-500 mb-2" />
                    <p>No data found. Ensure the ingestion script has run.</p>
                </div>
            )}
        </div>
    );
};

export default Macro;

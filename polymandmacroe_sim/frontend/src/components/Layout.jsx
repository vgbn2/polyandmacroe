import React from 'react';
import { Link, Outlet, useLocation } from 'react-router-dom';
import { LayoutDashboard, Globe, TrendingUp, Settings, Search } from 'lucide-react';

const SidebarItem = ({ to, icon: Icon, label, active }) => (
    <Link
        to={to}
        className={`flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors ${active ? 'bg-primary/20 text-primary border-r-2 border-primary' : 'text-gray-400 hover:bg-surface hover:text-white'
            }`}
    >
        <Icon size={20} />
        <span className="font-medium">{label}</span>
    </Link>
);

const Layout = () => {
    const location = useLocation();
    const isActive = (path) => location.pathname === path;

    return (
        <div className="flex h-screen bg-background text-white overflow-hidden">
            {/* Sidebar */}
            <aside className="w-64 border-r border-surface flex flex-col bg-background/50 backdrop-blur-md">
                <div className="p-6 border-b border-surface">
                    <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-emerald-400 bg-clip-text text-transparent">
                        Edge Terminal
                    </h1>
                </div>

                <nav className="flex-1 p-4 space-y-2">
                    <SidebarItem to="/" icon={LayoutDashboard} label="Dashboard" active={isActive('/')} />
                    <SidebarItem to="/macro" icon={Globe} label="Macro Grader" active={isActive('/macro')} />
                    <SidebarItem to="/polymarket" icon={TrendingUp} label="Edge Finder" active={isActive('/polymarket')} />
                </nav>

                <div className="p-4 border-t border-surface">
                    <SidebarItem to="/settings" icon={Settings} label="Settings" active={isActive('/settings')} />
                </div>
            </aside>

            {/* Main Content */}
            <main className="flex-1 flex flex-col min-w-0 overflow-hidden">
                {/* Header */}
                <header className="h-16 border-b border-surface flex items-center justify-between px-6 bg-background/95 backdrop-blur z-10">
                    <div className="flex items-center text-sm breadcrumbs text-gray-400">
                        <span className="text-white font-medium capitalize">{location.pathname === '/' ? 'Dashboard' : location.pathname.slice(1)}</span>
                    </div>

                    <div className="relative w-96">
                        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-500" size={16} />
                        <input
                            type="text"
                            placeholder="Search markets, indicators or assets..."
                            className="w-full bg-surface border border-gray-700 text-sm rounded-md pl-10 pr-4 py-2 focus:outline-none focus:border-primary transition-all placeholder-gray-500"
                        />
                    </div>
                </header>

                {/* Scrollable Canvas */}
                <div className="flex-1 overflow-auto p-6 scrollbar-hide">
                    <div className="max-w-7xl mx-auto">
                        <Outlet />
                    </div>
                </div>
            </main>
        </div>
    );
};

export default Layout;

import React from 'react';
import { NetworkStats as NetworkStatsType } from '../types/blockchain';
import { Activity, Globe, Hash, TrendingUp, Users, Zap } from 'lucide-react';

interface NetworkStatsProps {
  stats: NetworkStatsType;
  onSyncNodes: () => void;
}

export const NetworkStats: React.FC<NetworkStatsProps> = ({ stats, onSyncNodes }) => {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700 p-8 transition-colors duration-200">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-semibold text-gray-900 dark:text-white transition-colors duration-200">Network Stats</h2>
        <button
          onClick={onSyncNodes}
          className="bg-emerald-600 hover:bg-emerald-700 dark:bg-emerald-500 dark:hover:bg-emerald-600 text-white px-4 py-2 rounded-lg font-medium transition-colors duration-200 flex items-center gap-2"
        >
          <Globe className="w-4 h-4" />
          Sync All
        </button>
      </div>
      
      <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-4 transition-colors duration-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-blue-600 dark:text-blue-400 transition-colors duration-200">Nodes</p>
              <p className="text-2xl font-bold text-blue-900 dark:text-blue-100 transition-colors duration-200">{stats.totalNodes}</p>
            </div>
            <Users className="w-8 h-8 text-blue-500 dark:text-blue-400 transition-colors duration-200" />
          </div>
        </div>
        
        <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-xl p-4 transition-colors duration-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-emerald-600 dark:text-emerald-400 transition-colors duration-200">Online</p>
              <p className="text-2xl font-bold text-emerald-900 dark:text-emerald-100 transition-colors duration-200">{stats.onlineNodes}</p>
            </div>
            <Activity className="w-8 h-8 text-emerald-500 dark:text-emerald-400 transition-colors duration-200" />
          </div>
        </div>
        
        <div className="bg-purple-50 dark:bg-purple-900/20 rounded-xl p-4 transition-colors duration-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-purple-600 dark:text-purple-400 transition-colors duration-200">Transactions</p>
              <p className="text-2xl font-bold text-purple-900 dark:text-purple-100 transition-colors duration-200">{stats.totalTransactions}</p>
            </div>
            <TrendingUp className="w-8 h-8 text-purple-500 dark:text-purple-400 transition-colors duration-200" />
          </div>
        </div>
        
        <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-xl p-4 transition-colors duration-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-yellow-600 dark:text-yellow-400 transition-colors duration-200">Hash Rate</p>
              <p className="text-2xl font-bold text-yellow-900 dark:text-yellow-100 transition-colors duration-200">{stats.networkHashRate} H/s</p>
            </div>
            <Zap className="w-8 h-8 text-yellow-500 dark:text-yellow-400 transition-colors duration-200" />
          </div>
        </div>
        
        <div className="bg-red-50 dark:bg-red-900/20 rounded-xl p-4 transition-colors duration-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-red-600 dark:text-red-400 transition-colors duration-200">Difficulty</p>
              <p className="text-2xl font-bold text-red-900 dark:text-red-100 transition-colors duration-200">{stats.difficulty}</p>
            </div>
            <Hash className="w-8 h-8 text-red-500 dark:text-red-400 transition-colors duration-200" />
          </div>
        </div>
        
        <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-xl p-4 transition-colors duration-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-indigo-600 dark:text-indigo-400 transition-colors duration-200">Status</p>
              <p className="text-lg font-bold text-indigo-900 dark:text-indigo-100 transition-colors duration-200">
                {stats.onlineNodes > stats.totalNodes * 0.5 ? 'Healthy' : 'Degraded'}
              </p>
            </div>
            <div className={`w-8 h-8 rounded-full ${
              stats.onlineNodes > stats.totalNodes * 0.5 ? 'bg-emerald-500' : 'bg-red-500'
            }`} />
          </div>
        </div>
      </div>
    </div>
  );
};
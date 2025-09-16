
# System Monitoring and Health Checks for MarketPulse
import psutil
import time
import json
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    active_connections: int

class PerformanceMonitor:
    """Monitor system performance and health"""

    def __init__(self, history_limit: int = 100):
        self.history_limit = history_limit
        self.metrics_history: List[SystemMetrics] = []
        self.alerts_triggered: List[Dict] = []

    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        metrics = SystemMetrics(
            timestamp=datetime.now().isoformat(),
            cpu_percent=psutil.cpu_percent(interval=1),
            memory_percent=memory.percent,
            memory_available_gb=memory.available / (1024**3),
            disk_usage_percent=disk.percent,
            active_connections=len(psutil.net_connections())
        )

        # Store in history
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.history_limit:
            self.metrics_history.pop(0)

        return metrics

    def check_system_health(self) -> Dict[str, Any]:
        """Perform comprehensive system health check"""
        metrics = self.collect_system_metrics()

        health_status = {
            'overall_health': 'HEALTHY',
            'issues': [],
            'warnings': [],
            'metrics': asdict(metrics),
            'recommendations': []
        }

        # CPU Health Check
        if metrics.cpu_percent > 90:
            health_status['issues'].append('High CPU usage detected')
            health_status['overall_health'] = 'CRITICAL'
        elif metrics.cpu_percent > 70:
            health_status['warnings'].append('Elevated CPU usage')
            if health_status['overall_health'] == 'HEALTHY':
                health_status['overall_health'] = 'WARNING'

        # Memory Health Check
        if metrics.memory_percent > 90:
            health_status['issues'].append('High memory usage detected')
            health_status['overall_health'] = 'CRITICAL'
            health_status['recommendations'].append('Consider restarting services or increasing memory')
        elif metrics.memory_percent > 75:
            health_status['warnings'].append('Elevated memory usage')
            if health_status['overall_health'] == 'HEALTHY':
                health_status['overall_health'] = 'WARNING'

        # Disk Health Check
        if metrics.disk_usage_percent > 95:
            health_status['issues'].append('Disk space critically low')
            health_status['overall_health'] = 'CRITICAL'
        elif metrics.disk_usage_percent > 85:
            health_status['warnings'].append('Disk space running low')
            if health_status['overall_health'] == 'HEALTHY':
                health_status['overall_health'] = 'WARNING'

        # Memory availability check
        if metrics.memory_available_gb < 1:
            health_status['warnings'].append('Low available memory')
            health_status['recommendations'].append('Close unnecessary applications')

        return health_status

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from recent metrics"""
        if not self.metrics_history:
            return {'status': 'No metrics available'}

        recent_metrics = self.metrics_history[-10:]  # Last 10 readings

        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        min_memory_available = min(m.memory_available_gb for m in recent_metrics)

        return {
            'metrics_collected': len(self.metrics_history),
            'recent_average_cpu': round(avg_cpu, 2),
            'recent_average_memory': round(avg_memory, 2),
            'minimum_available_memory_gb': round(min_memory_available, 2),
            'current_health': self.check_system_health()['overall_health']
        }

class TradingSystemMonitor:
    """Monitor MarketPulse trading system components"""

    def __init__(self):
        self.component_status = {}
        self.performance_metrics = {}
        self.error_counts = {}

    def register_component(self, component_name: str):
        """Register a system component for monitoring"""
        self.component_status[component_name] = {
            'status': 'UNKNOWN',
            'last_check': None,
            'uptime_start': datetime.now(),
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0
        }

    def update_component_status(self, component_name: str, status: str, 
                              operation_success: bool = True):
        """Update component status and performance metrics"""
        if component_name not in self.component_status:
            self.register_component(component_name)

        component = self.component_status[component_name]
        component['status'] = status
        component['last_check'] = datetime.now()
        component['total_operations'] += 1

        if operation_success:
            component['successful_operations'] += 1
        else:
            component['failed_operations'] += 1
            self.error_counts[component_name] = self.error_counts.get(component_name, 0) + 1

    def get_component_health(self) -> Dict[str, Any]:
        """Get health status of all registered components"""
        health_report = {
            'overall_system_health': 'HEALTHY',
            'components': {},
            'critical_issues': [],
            'warnings': []
        }

        for component_name, status in self.component_status.items():
            success_rate = 0
            if status['total_operations'] > 0:
                success_rate = status['successful_operations'] / status['total_operations']

            component_health = {
                'status': status['status'],
                'success_rate': round(success_rate * 100, 2),
                'total_operations': status['total_operations'],
                'error_count': self.error_counts.get(component_name, 0),
                'uptime_hours': (datetime.now() - status['uptime_start']).total_seconds() / 3600
            }

            # Determine component health
            if success_rate < 0.5 and status['total_operations'] > 5:
                component_health['health'] = 'CRITICAL'
                health_report['critical_issues'].append(f"{component_name} has low success rate")
                health_report['overall_system_health'] = 'CRITICAL'
            elif success_rate < 0.8 and status['total_operations'] > 3:
                component_health['health'] = 'WARNING'
                health_report['warnings'].append(f"{component_name} has degraded performance")
                if health_report['overall_system_health'] == 'HEALTHY':
                    health_report['overall_system_health'] = 'WARNING'
            else:
                component_health['health'] = 'HEALTHY'

            health_report['components'][component_name] = component_health

        return health_report

    def save_health_report(self, filepath: str = 'system_health.json'):
        """Save current health report to file"""
        health_report = {
            'timestamp': datetime.now().isoformat(),
            'system_health': self.get_component_health(),
            'performance_summary': system_monitor.get_performance_summary() if 'system_monitor' in globals() else None
        }

        with open(filepath, 'w') as f:
            json.dump(health_report, f, indent=2, default=str)

# Global monitoring instances
system_monitor = PerformanceMonitor()
trading_monitor = TradingSystemMonitor()

# Register core MarketPulse components
trading_monitor.register_component('technical_analysis')
trading_monitor.register_component('fundamental_analysis')
trading_monitor.register_component('sentiment_analysis')
trading_monitor.register_component('risk_management')
trading_monitor.register_component('ai_framework')

def quick_health_check() -> Dict[str, Any]:
    """Perform quick system and component health check"""
    return {
        'system': system_monitor.check_system_health(),
        'trading_components': trading_monitor.get_component_health(),
        'timestamp': datetime.now().isoformat()
    }

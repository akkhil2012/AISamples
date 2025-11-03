import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def create_test_data():
    """Generate multiple test CSV files for anomaly detection testing"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create output directory
    if not os.path.exists('test_data'):
        os.makedirs('test_data')
    
    print("Generating test datasets...")
    
    # Test Case 1: Web Server Logs
    generate_web_server_logs()
    
    # Test Case 2: System Performance Logs
    generate_system_performance_logs()
    
    # Test Case 3: Application Error Logs
    generate_application_logs()
    
    # Test Case 4: Network Traffic Logs
    generate_network_logs()
    
    # Test Case 5: Database Transaction Logs
    generate_database_logs()
    
    print("All test datasets generated successfully!")
    print("\nGenerated files:")
    for file in os.listdir('test_data'):
        if file.endswith('.csv'):
            print(f"  - test_data/{file}")

def generate_web_server_logs():
    """Generate web server access logs with anomalies"""
    print("Creating web server logs...")
    
    n_samples = 5000
    start_date = datetime(2024, 1, 1)
    
    # Normal data distribution
    data = {
        'timestamp': [start_date + timedelta(minutes=i/10) for i in range(n_samples)],
        'response_time_ms': np.random.lognormal(mean=4, sigma=0.5, size=n_samples),  # ~55ms average
        'status_code': np.random.choice([200, 301, 404, 500, 503], n_samples, 
                                       p=[0.85, 0.05, 0.07, 0.02, 0.01]),
        'request_size_bytes': np.random.exponential(scale=1024, size=n_samples),
        'response_size_bytes': np.random.lognormal(mean=8, sigma=1, size=n_samples),
        'user_agent': np.random.choice(['Chrome', 'Firefox', 'Safari', 'Edge', 'Bot'], n_samples,
                                      p=[0.45, 0.25, 0.15, 0.10, 0.05]),
        'http_method': np.random.choice(['GET', 'POST', 'PUT', 'DELETE'], n_samples,
                                       p=[0.70, 0.20, 0.07, 0.03]),
        'cpu_usage_percent': np.random.normal(35, 10, n_samples),
        'memory_usage_percent': np.random.normal(60, 15, n_samples),
        'concurrent_users': np.random.poisson(lam=50, size=n_samples)
    }
    
    # Inject anomalies (10% of data)
    anomaly_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    
    for idx in anomaly_indices:
        anomaly_type = np.random.choice(['slow_response', 'high_error', 'resource_spike', 'bot_attack'])
        
        if anomaly_type == 'slow_response':
            data['response_time_ms'][idx] = np.random.uniform(5000, 15000)  # 5-15 seconds
            data['cpu_usage_percent'][idx] = np.random.uniform(80, 95)
            
        elif anomaly_type == 'high_error':
            data['status_code'][idx] = np.random.choice([500, 503, 504])
            data['response_time_ms'][idx] = np.random.uniform(1000, 3000)
            
        elif anomaly_type == 'resource_spike':
            data['cpu_usage_percent'][idx] = np.random.uniform(90, 100)
            data['memory_usage_percent'][idx] = np.random.uniform(90, 100)
            data['response_time_ms'][idx] = np.random.uniform(2000, 8000)
            
        elif anomaly_type == 'bot_attack':
            data['concurrent_users'][idx] = np.random.uniform(200, 500)
            data['request_size_bytes'][idx] = np.random.uniform(10000, 50000)
            data['user_agent'][idx] = 'Bot'
    
    df = pd.DataFrame(data)
    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df.to_csv('test_data/web_server_logs.csv', index=False)
    print(f"  Generated: web_server_logs.csv ({len(df)} rows)")

def generate_system_performance_logs():
    """Generate system performance monitoring logs"""
    print("Creating system performance logs...")
    
    n_samples = 3000
    
    # Simulate 24-hour cycle effects
    hours = np.tile(np.arange(24), n_samples // 24 + 1)[:n_samples]
    
    # Base values with daily patterns
    base_cpu = 30 + 20 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 5, n_samples)
    base_memory = 50 + 10 * np.sin(2 * np.pi * (hours - 6) / 24) + np.random.normal(0, 8, n_samples)
    
    data = {
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='5T'),
        'cpu_usage_percent': np.clip(base_cpu, 0, 100),
        'memory_usage_percent': np.clip(base_memory, 0, 100),
        'disk_io_read_mbps': np.random.exponential(scale=10, size=n_samples),
        'disk_io_write_mbps': np.random.exponential(scale=5, size=n_samples),
        'network_in_mbps': np.random.lognormal(mean=2, sigma=1, size=n_samples),
        'network_out_mbps': np.random.lognormal(mean=2, sigma=1, size=n_samples),
        'active_processes': np.random.poisson(lam=150, size=n_samples),
        'temperature_celsius': np.random.normal(45, 8, n_samples),
        'load_average': np.random.exponential(scale=2, size=n_samples),
        'server_id': np.random.choice(['SRV-001', 'SRV-002', 'SRV-003', 'SRV-004'], n_samples)
    }
    
    # Inject system anomalies
    anomaly_indices = np.random.choice(n_samples, size=int(0.08 * n_samples), replace=False)
    
    for idx in anomaly_indices:
        anomaly_type = np.random.choice(['cpu_spike', 'memory_leak', 'disk_thrash', 'thermal_issue'])
        
        if anomaly_type == 'cpu_spike':
            data['cpu_usage_percent'][idx] = np.random.uniform(95, 100)
            data['load_average'][idx] = np.random.uniform(8, 15)
            data['temperature_celsius'][idx] = np.random.uniform(70, 85)
            
        elif anomaly_type == 'memory_leak':
            data['memory_usage_percent'][idx] = np.random.uniform(85, 98)
            data['active_processes'][idx] = np.random.uniform(300, 500)
            
        elif anomaly_type == 'disk_thrash':
            data['disk_io_read_mbps'][idx] = np.random.uniform(100, 200)
            data['disk_io_write_mbps'][idx] = np.random.uniform(50, 100)
            data['cpu_usage_percent'][idx] = np.random.uniform(70, 90)
            
        elif anomaly_type == 'thermal_issue':
            data['temperature_celsius'][idx] = np.random.uniform(80, 95)
            data['cpu_usage_percent'][idx] = np.random.uniform(20, 40)  # Thermal throttling
    
    df = pd.DataFrame(data)
    df.to_csv('test_data/system_performance_logs.csv', index=False)
    print(f"  Generated: system_performance_logs.csv ({len(df)} rows)")

def generate_application_logs():
    """Generate application logs with various error patterns"""
    print("Creating application logs...")
    
    n_samples = 4000
    
    data = {
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='2T'),
        'log_level': np.random.choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], n_samples,
                                    p=[0.30, 0.50, 0.15, 0.04, 0.01]),
        'module_name': np.random.choice(['auth', 'database', 'api', 'ui', 'cache', 'payment'], n_samples),
        'execution_time_ms': np.random.lognormal(mean=3, sigma=0.8, size=n_samples),
        'memory_allocated_mb': np.random.exponential(scale=50, size=n_samples),
        'database_connections': np.random.poisson(lam=10, size=n_samples),
        'cache_hit_rate': np.random.beta(a=8, b=2, size=n_samples) * 100,  # Mostly high hit rates
        'queue_size': np.random.poisson(lam=5, size=n_samples),
        'error_count': np.random.poisson(lam=0.5, size=n_samples),
        'user_session_count': np.random.poisson(lam=100, size=n_samples),
        'application_version': np.random.choice(['v1.2.1', 'v1.2.2', 'v1.3.0'], n_samples,
                                              p=[0.20, 0.30, 0.50])
    }
    
    # Inject application anomalies
    anomaly_indices = np.random.choice(n_samples, size=int(0.12 * n_samples), replace=False)
    
    for idx in anomaly_indices:
        anomaly_type = np.random.choice(['memory_leak', 'database_timeout', 'cache_miss', 'queue_backup'])
        
        if anomaly_type == 'memory_leak':
            data['memory_allocated_mb'][idx] = np.random.uniform(500, 1000)
            data['execution_time_ms'][idx] = np.random.uniform(5000, 15000)
            data['log_level'][idx] = np.random.choice(['WARNING', 'ERROR'])
            
        elif anomaly_type == 'database_timeout':
            data['execution_time_ms'][idx] = np.random.uniform(30000, 60000)  # 30-60 seconds
            data['database_connections'][idx] = np.random.uniform(50, 100)
            data['log_level'][idx] = 'ERROR'
            data['module_name'][idx] = 'database'
            
        elif anomaly_type == 'cache_miss':
            data['cache_hit_rate'][idx] = np.random.uniform(0, 20)  # Very low hit rate
            data['execution_time_ms'][idx] = np.random.uniform(2000, 8000)
            data['module_name'][idx] = 'cache'
            
        elif anomaly_type == 'queue_backup':
            data['queue_size'][idx] = np.random.uniform(100, 500)
            data['execution_time_ms'][idx] = np.random.uniform(10000, 25000)
            data['log_level'][idx] = np.random.choice(['WARNING', 'ERROR'])
    
    df = pd.DataFrame(data)
    df.to_csv('test_data/application_logs.csv', index=False)
    print(f"  Generated: application_logs.csv ({len(df)} rows)")

def generate_network_logs():
    """Generate network traffic logs with security anomalies"""
    print("Creating network traffic logs...")
    
    n_samples = 6000
    
    # Generate IP addresses
    def generate_ip():
        return f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"
    
    data = {
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='30S'),
        'source_ip': [generate_ip() for _ in range(n_samples)],
        'destination_port': np.random.choice([80, 443, 22, 21, 25, 53, 3306, 5432], n_samples,
                                           p=[0.35, 0.30, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05]),
        'packet_size_bytes': np.random.lognormal(mean=7, sigma=1.5, size=n_samples),
        'packets_per_second': np.random.exponential(scale=10, size=n_samples),
        'protocol': np.random.choice(['TCP', 'UDP', 'ICMP'], n_samples, p=[0.70, 0.25, 0.05]),
        'connection_duration_sec': np.random.exponential(scale=30, size=n_samples),
        'bytes_transferred': np.random.lognormal(mean=10, sigma=2, size=n_samples),
        'connection_state': np.random.choice(['ESTABLISHED', 'SYN_SENT', 'CLOSED', 'TIME_WAIT'], n_samples,
                                           p=[0.60, 0.15, 0.15, 0.10]),
        'country_code': np.random.choice(['US', 'CA', 'GB', 'DE', 'CN', 'RU', 'BR'], n_samples,
                                        p=[0.40, 0.15, 0.10, 0.10, 0.10, 0.08, 0.07])
    }
    
    # Inject network anomalies
    anomaly_indices = np.random.choice(n_samples, size=int(0.15 * n_samples), replace=False)
    
    for idx in anomaly_indices:
        anomaly_type = np.random.choice(['ddos_attack', 'port_scan', 'data_exfiltration', 'brute_force'])
        
        if anomaly_type == 'ddos_attack':
            data['packets_per_second'][idx] = np.random.uniform(1000, 5000)
            data['packet_size_bytes'][idx] = np.random.uniform(64, 128)  # Small packets
            data['connection_duration_sec'][idx] = np.random.uniform(0.1, 1)
            data['protocol'][idx] = 'UDP'
            
        elif anomaly_type == 'port_scan':
            data['destination_port'][idx] = np.random.randint(1, 65535)  # Random ports
            data['packets_per_second'][idx] = np.random.uniform(50, 200)
            data['connection_duration_sec'][idx] = np.random.uniform(0.1, 0.5)
            data['connection_state'][idx] = 'SYN_SENT'
            
        elif anomaly_type == 'data_exfiltration':
            data['bytes_transferred'][idx] = np.random.uniform(1e8, 1e10)  # 100MB-10GB
            data['connection_duration_sec'][idx] = np.random.uniform(600, 3600)  # 10min-1hr
            data['destination_port'][idx] = np.random.choice([80, 443])
            
        elif anomaly_type == 'brute_force':
            data['destination_port'][idx] = 22  # SSH
            data['packets_per_second'][idx] = np.random.uniform(10, 50)
            data['connection_duration_sec'][idx] = np.random.uniform(0.1, 2)
            data['connection_state'][idx] = 'CLOSED'
    
    df = pd.DataFrame(data)
    df.to_csv('test_data/network_logs.csv', index=False)
    print(f"  Generated: network_logs.csv ({len(df)} rows)")

def generate_database_logs():
    """Generate database transaction"""

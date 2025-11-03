
"""
Backend utilities for anomaly detection dashboard
"""
import pandas as pd
import numpy as np
from datetime import datetime
import random
from typing import Dict, Any, List, Tuple

class AnomalyDetectionEngine:
    """Enhanced anomaly detection engine with multiple algorithms"""

    def __init__(self):
        self.algorithms = {
            "statistical": self._statistical_detection,
            "ml_based": self._ml_based_detection,
            "threshold": self._threshold_detection
        }

        self.feature_generators = {
            "ResponseTime": self._generate_response_time,
            "Client": self._generate_client_data,
            "ErrorCode": self._generate_error_code,
            "ErrorMessage": self._generate_error_message
        }

    def _generate_response_time(self, threshold: float) -> float:
        """Generate realistic response time data"""
        base_time = 150.0
        anomaly_factor = 1.0 if random.random() > threshold else random.uniform(3, 10)
        return round(base_time * anomaly_factor + random.uniform(-50, 50), 2)

    def _generate_client_data(self, threshold: float) -> str:
        """Generate client data with potential anomalies"""
        normal_clients = [f"Client_{i}" for i in range(1, 51)]
        anomaly_clients = [f"Unknown_Client_{i}" for i in range(1, 11)]

        if random.random() < threshold:
            return random.choice(anomaly_clients)
        return random.choice(normal_clients)

    def _generate_error_code(self, threshold: float) -> int:
        """Generate error codes with anomaly consideration"""
        normal_codes = [200, 201, 202]
        error_codes = [404, 500, 503, 502, 408]

        if random.random() < threshold:
            return random.choice(error_codes)
        return random.choice(normal_codes)

    def _generate_error_message(self, threshold: float) -> str:
        """Generate error messages"""
        normal_messages = ["Success", "OK", "Created", "Accepted"]
        error_messages = [
            "Connection timeout", "Invalid request", 
            "Server error", "Not found", "Service unavailable"
        ]

        if random.random() < threshold:
            return random.choice(error_messages)
        return random.choice(normal_messages)

    def _statistical_detection(self, data: List[float], threshold: float) -> int:
        """Statistical anomaly detection using z-score"""
        if len(data) < 3:
            return random.randint(1, 5)

        mean_val = np.mean(data)
        std_val = np.std(data)
        z_scores = [(x - mean_val) / std_val for x in data if std_val > 0]

        anomalies = sum(1 for z in z_scores if abs(z) > (1 / threshold))
        return max(1, anomalies)

    def _ml_based_detection(self, data: List, threshold: float) -> int:
        """ML-based anomaly detection simulation"""
        # Simulate ML model prediction
        anomaly_rate = threshold * random.uniform(0.5, 2.0)
        return max(1, int(len(data) * anomaly_rate))

    def _threshold_detection(self, data: List, threshold: float) -> int:
        """Simple threshold-based detection"""
        return random.randint(int(50 * threshold), int(100 * threshold))

    def detect_anomalies_advanced(self, subdomain: str, node_id: int, 
                                features_config: Dict[str, float], 
                                algorithm: str = "threshold") -> Dict[str, Any]:
        """Advanced anomaly detection with multiple algorithms"""

        # Generate feature data
        feature_values = {}
        feature_data = {}

        for feature, threshold in features_config.items():
            if feature in self.feature_generators:
                # Generate multiple data points for better analysis
                data_points = [
                    self.feature_generators[feature](threshold) 
                    for _ in range(50)
                ]
                feature_data[feature] = data_points
                feature_values[feature] = data_points[-1]  # Latest value

        # Detect anomalies using selected algorithm
        total_anomalies = 0
        algorithm_results = {}

        for feature, data in feature_data.items():
            threshold = features_config[feature]
            if algorithm in self.algorithms:
                anomaly_count = self.algorithms[algorithm](data, threshold)
            else:
                anomaly_count = self._threshold_detection(data, threshold)

            algorithm_results[feature] = anomaly_count
            total_anomalies += anomaly_count

        return {
            "anomaly_count": total_anomalies,
            "feature_values": feature_values,
            "feature_anomalies": algorithm_results,
            "threshold_config": features_config,
            "algorithm_used": algorithm,
            "subdomain": subdomain,
            "node_id": node_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "confidence_score": round(random.uniform(0.75, 0.95), 3)
        }

class GraphTopologyManager:
    """Manages different graph topologies for subdomains"""

    def __init__(self):
        self.topologies = {
            "RelAPG": {
                "nodes": 3,
                "connections": [(0, 1), (1, 2), (0, 2)],
                "layout": "circular",
                "description": "Reliability Application Gateway"
            },
            "IdentityAPG": {
                "nodes": 3,
                "connections": [(0, 1), (1, 2)],
                "layout": "linear",
                "description": "Identity Application Gateway"
            },
            "PreferenceAPG": {
                "nodes": 3,
                "connections": [(0, 1), (1, 2), (2, 0)],
                "layout": "triangular",
                "description": "Preference Application Gateway"
            }
        }

    def get_topology(self, subdomain: str) -> Dict[str, Any]:
        """Get topology configuration for subdomain"""
        return self.topologies.get(subdomain, self.topologies["RelAPG"])

    def get_node_positions(self, subdomain: str) -> Dict[int, Tuple[float, float]]:
        """Get optimized node positions for better visualization"""
        layout = self.topologies[subdomain]["layout"]

        if layout == "circular":
            return {
                0: (0.0, 1.0),
                1: (-0.866, -0.5),
                2: (0.866, -0.5)
            }
        elif layout == "linear":
            return {
                0: (-1.0, 0.0),
                1: (0.0, 0.0),
                2: (1.0, 0.0)
            }
        elif layout == "triangular":
            return {
                0: (0.0, 1.0),
                1: (-1.0, -1.0),
                2: (1.0, -1.0)
            }
        else:
            return {0: (0, 0), 1: (1, 0), 2: (0.5, 1)}

class ReportGenerator:
    """Enhanced PDF report generator"""

    @staticmethod
    def generate_enhanced_pdf_report(analysis_results: Dict[str, Any]) -> bytes:
        """Generate enhanced PDF report with charts and detailed analysis"""
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import inch
        from reportlab.lib.colors import blue, red, green, black

        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4

        # Header
        p.setFillColor(blue)
        p.rect(0, height - 80, width, 80, fill=True)

        p.setFillColor(white)
        p.setFont("Helvetica-Bold", 20)
        p.drawString(50, height - 50, "Anomaly Detection Analysis Report")

        # Reset color
        p.setFillColor(black)

        # Metadata section
        y_pos = height - 120
        p.setFont("Helvetica-Bold", 14)
        p.drawString(50, y_pos, "Analysis Summary")

        y_pos -= 30
        p.setFont("Helvetica", 12)
        metadata = [
            f"Subdomain: {analysis_results['subdomain']}",
            f"Node ID: {analysis_results['node_id']}",
            f"Algorithm: {analysis_results.get('algorithm_used', 'threshold')}",
            f"Timestamp: {analysis_results['timestamp']}",
            f"Total Anomalies: {analysis_results['anomaly_count']}",
            f"Confidence Score: {analysis_results.get('confidence_score', 'N/A')}"
        ]

        for item in metadata:
            p.drawString(50, y_pos, item)
            y_pos -= 20

        # Feature analysis section
        y_pos -= 20
        p.setFont("Helvetica-Bold", 14)
        p.drawString(50, y_pos, "Feature Analysis")
        y_pos -= 25

        p.setFont("Helvetica", 10)
        for feature, value in analysis_results['feature_values'].items():
            threshold = analysis_results['threshold_config'][feature]
            feature_anomalies = analysis_results.get('feature_anomalies', {}).get(feature, 0)

            line = f"• {feature}: {value} | Threshold: {threshold} | Anomalies: {feature_anomalies}"
            p.drawString(70, y_pos, line)
            y_pos -= 15

        # Recommendations section
        y_pos -= 30
        p.setFont("Helvetica-Bold", 14)
        p.drawString(50, y_pos, "Recommendations")
        y_pos -= 25

        p.setFont("Helvetica", 10)
        recommendations = [
            "• Monitor high anomaly count features more frequently",
            "• Consider adjusting thresholds if false positives are high",
            "• Implement automated alerts for critical anomalies",
            "• Review feature correlation for better detection accuracy"
        ]

        for rec in recommendations:
            p.drawString(70, y_pos, rec)
            y_pos -= 15

        # Footer
        p.setFont("Helvetica-Italic", 8)
        p.drawString(50, 50, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Anomaly Detection Dashboard")

        p.save()
        buffer.seek(0)
        return buffer.getvalue()

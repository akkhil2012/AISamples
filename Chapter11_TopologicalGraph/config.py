
"""
Configuration settings for the anomaly detection dashboard
"""

class DashboardConfig:
    """Configuration class for dashboard settings"""

    # SubDomain configurations
    SUBDOMAINS = {
        "RelAPG": {
            "name": "Reliability Application Gateway",
            "description": "Monitors reliability metrics and performance indicators",
            "default_features": ["ResponseTime", "ErrorCode"],
            "color_scheme": "#1f77b4"
        },
        "IdentityAPG": {
            "name": "Identity Application Gateway", 
            "description": "Handles authentication and identity management",
            "default_features": ["Client", "ErrorMessage"],
            "color_scheme": "#ff7f0e"
        },
        "PreferenceAPG": {
            "name": "Preference Application Gateway",
            "description": "Manages user preferences and settings",
            "default_features": ["ResponseTime", "Client"],
            "color_scheme": "#2ca02c"
        }
    }

    # Feature configurations
    FEATURES = {
        "ResponseTime": {
            "type": "numeric",
            "unit": "ms",
            "description": "API response time in milliseconds",
            "normal_range": (50, 200)
        },
        "Client": {
            "type": "categorical",
            "description": "Client identifier or source",
            "categories": ["Web", "Mobile", "API", "Unknown"]
        },
        "ErrorCode": {
            "type": "numeric",
            "description": "HTTP status or error codes",
            "normal_codes": [200, 201, 202],
            "error_codes": [400, 401, 404, 500, 502, 503]
        },
        "ErrorMessage": {
            "type": "text",
            "description": "Error message content",
            "categories": ["Timeout", "Invalid", "Server Error", "Not Found"]
        }
    }

    # Threshold options
    THRESHOLD_OPTIONS = [0.01, 0.05, 0.10, 0.20]

    # Algorithm options
    ALGORITHMS = {
        "threshold": "Simple threshold-based detection",
        "statistical": "Statistical analysis using z-scores",
        "ml_based": "Machine learning based detection"
    }

    # UI Configuration
    UI_CONFIG = {
        "page_title": "Anomaly Detection Dashboard",
        "page_icon": "🔍",
        "layout": "wide",
        "theme": {
            "primary_color": "#1f77b4",
            "background_color": "#ffffff",
            "secondary_background_color": "#f0f2f6"
        }
    }

    # Export settings
    EXPORT_CONFIG = {
        "pdf_settings": {
            "page_size": "A4",
            "font_family": "Helvetica",
            "include_charts": True
        },
        "csv_settings": {
            "delimiter": ",",
            "include_metadata": True
        }
    }

"""
Main entry point for Oxide Descriptor Analysis
"""

import argparse
import json
import logging
from pathlib import Path

from oxide_analysis_general.oxide_analysis_general import OxideAnalysis
from oxide_analysis_general.config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)

def main():
    """Main function to run the analysis"""
    parser = argparse.ArgumentParser(description='Oxide Descriptor Analysis')
    parser.add_argument('--data', type=str, default='oxide_descriptors_integrated.xlsx',
                       help='Path to Excel data file')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to JSON configuration file')
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    if args.config:
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
            config.update(custom_config)

    analyzer = OxideAnalysis(config)
    results = analyzer.run_analysis(args.data)

    logger.info("Analysis completed successfully!")
    return results

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Test script for data merging functionality.

This script tests the data merging, alignment, and comparison functions
from the data_transformer module.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.utils.data_transformer import (
    merge_stock_data,
    merge_from_multiple_sources,
    align_time_series,
    resample_and_align,
    normalize_for_comparison,
    DataTransformationError
)

class TestDataMerging(unittest.TestCase):
    """Test cases for data merging functionality."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample DataFrames for testing
        dates1 = pd.date_range(start='2020-01-01', periods=10, freq='D')
        dates2 = pd.date_range(start='2020-01-05', periods=10, freq='D')
        
        # DataFrame for ticker1
        self.df1 = pd.DataFrame({
            'ticker': ['AAPL'] * 10,
            'open': np.linspace(100, 110, 10),
            'high': np.linspace(105, 115, 10),
            'low': np.linspace(95, 105, 10),
            'close': np.linspace(102, 112, 10),
            'volume': np.linspace(1000000, 1100000, 10)
        }, index=dates1)
        
        # DataFrame for ticker2
        self.df2 = pd.DataFrame({
            'ticker': ['MSFT'] * 10,
            'open': np.linspace(200, 210, 10),
            'high': np.linspace(205, 215, 10),
            'low': np.linspace(195, 205, 10),
            'close': np.linspace(202, 212, 10),
            'volume': np.linspace(2000000, 2100000, 10)
        }, index=dates2)
        
        # DataFrame for ticker1 from a different source
        self.df1_source2 = pd.DataFrame({
            'ticker': ['AAPL'] * 10,
            'open': np.linspace(101, 111, 10),
            'high': np.linspace(106, 116, 10),
            'low': np.linspace(96, 106, 10),
            'close': np.linspace(103, 113, 10),
            'volume': np.linspace(1010000, 1110000, 10)
        }, index=dates1)
    
    def test_align_time_series_inner(self):
        """Test aligning time series with inner join."""
        aligned_dfs = align_time_series([self.df1, self.df2], method='inner', fill_method=None)
        
        # Check that we have two DataFrames
        self.assertEqual(len(aligned_dfs), 2)
        
        # Check that they have the same dates (inner join should have 6 dates)
        self.assertEqual(len(aligned_dfs[0]), 6)
        self.assertEqual(len(aligned_dfs[1]), 6)
        
        # Check that the dates are the same
        pd.testing.assert_index_equal(aligned_dfs[0].index, aligned_dfs[1].index)
        
        # Check that the first date is 2020-01-05 (the start of df2)
        self.assertEqual(aligned_dfs[0].index[0], pd.Timestamp('2020-01-05'))
    
    def test_align_time_series_outer(self):
        """Test aligning time series with outer join."""
        aligned_dfs = align_time_series([self.df1, self.df2], method='outer', fill_method=None)
        
        # Check that we have two DataFrames
        self.assertEqual(len(aligned_dfs), 2)
        
        # Check that they have the same dates (outer join should have 14 dates)
        self.assertEqual(len(aligned_dfs[0]), 14)
        self.assertEqual(len(aligned_dfs[1]), 14)
        
        # Check that the dates are the same
        pd.testing.assert_index_equal(aligned_dfs[0].index, aligned_dfs[1].index)
        
        # Check that the first date is 2020-01-01 (the start of df1)
        self.assertEqual(aligned_dfs[0].index[0], pd.Timestamp('2020-01-01'))
        
        # Check that NaN values exist in the appropriate places
        self.assertTrue(pd.isna(aligned_dfs[1].loc['2020-01-01', 'close']))
        self.assertTrue(pd.isna(aligned_dfs[0].loc['2020-01-14', 'close']))
    
    def test_align_time_series_with_fill(self):
        """Test aligning time series with filling missing values."""
        aligned_dfs = align_time_series([self.df1, self.df2], method='outer', fill_method='ffill')
        
        # Check that NaN values are filled after the first data point
        # For df2, the first data point is 2020-01-05, so 2020-01-06 should be filled if missing
        if '2020-01-06' in aligned_dfs[1].index and pd.isna(aligned_dfs[1].loc['2020-01-06', 'close']):
            self.assertFalse(pd.isna(aligned_dfs[1].loc['2020-01-06', 'close']))
        
        # The first few dates in df2 should still be NaN (can't forward fill before data exists)
        self.assertTrue(pd.isna(aligned_dfs[1].loc['2020-01-01', 'close']))
    
    def test_normalize_for_comparison_percent_change(self):
        """Test normalizing data using percent change method."""
        normalized_dfs = normalize_for_comparison([self.df1, self.df2], column='close', method='percent_change')
        
        # Check that we have two DataFrames
        self.assertEqual(len(normalized_dfs), 2)
        
        # Check that each DataFrame has a normalized column
        self.assertIn('close_normalized', normalized_dfs[0].columns)
        self.assertIn('close_normalized', normalized_dfs[1].columns)
        
        # Check that the first value is 0 (percent change from first value)
        self.assertEqual(normalized_dfs[0]['close_normalized'].iloc[0], 0.0)
        self.assertEqual(normalized_dfs[1]['close_normalized'].iloc[0], 0.0)
        
        # Check that the last value is approximately 9.8% (for a linear increase of 10 points from 102)
        self.assertAlmostEqual(normalized_dfs[0]['close_normalized'].iloc[-1], 9.8, delta=0.1)
    
    def test_normalize_for_comparison_min_max(self):
        """Test normalizing data using min-max method."""
        normalized_dfs = normalize_for_comparison([self.df1, self.df2], column='close', method='min-max')
        
        # Check that the min value is 0 and max value is 1
        self.assertEqual(normalized_dfs[0]['close_normalized'].min(), 0.0)
        self.assertEqual(normalized_dfs[0]['close_normalized'].max(), 1.0)
        self.assertEqual(normalized_dfs[1]['close_normalized'].min(), 0.0)
        self.assertEqual(normalized_dfs[1]['close_normalized'].max(), 1.0)
    
    def test_resample_and_align(self):
        """Test resampling and aligning multiple DataFrames."""
        # Create daily data with some gaps
        dates1 = pd.date_range(start='2020-01-01', periods=30, freq='D')
        dates2 = pd.date_range(start='2020-01-10', periods=30, freq='D')
        
        df1 = pd.DataFrame({
            'ticker': ['AAPL'] * 30,
            'close': np.linspace(100, 130, 30)
        }, index=dates1)
        
        df2 = pd.DataFrame({
            'ticker': ['MSFT'] * 30,
            'close': np.linspace(200, 230, 30)
        }, index=dates2)
        
        # Resample to weekly and align
        resampled_dfs = resample_and_align([df1, df2], freq='W', method='outer', fill_method='ffill')
        
        # Check that we have two DataFrames
        self.assertEqual(len(resampled_dfs), 2)
        
        # Check that they have the same dates
        pd.testing.assert_index_equal(resampled_dfs[0].index, resampled_dfs[1].index)
        
        # Check that the frequency is weekly-like (different pandas versions might format this differently)
        # Just check that we have fewer points than the original daily data
        self.assertTrue(len(resampled_dfs[0]) < len(df1))
        
        # Check that the number of weeks is correct (should be around 8-9 weeks)
        self.assertTrue(5 <= len(resampled_dfs[0]) <= 10)
    
    def test_normalize_with_base_date(self):
        """Test normalizing with a specific base date."""
        # Use a date in the middle of the series
        base_date = '2020-01-05'
        
        normalized_dfs = normalize_for_comparison(
            [self.df1, self.df2], 
            column='close', 
            method='percent_change',
            base_date=base_date
        )
        
        # Find the index of the base date in df1
        base_idx = self.df1.index.get_indexer([pd.Timestamp(base_date)])[0]
        
        # The value at the base date should be 0
        self.assertEqual(normalized_dfs[0]['close_normalized'].iloc[base_idx], 0.0)
        
        # For df2, the base date might not exist, so it should use the closest date after
        # In this case, 2020-01-05 is the first date in df2
        self.assertEqual(normalized_dfs[1]['close_normalized'].iloc[0], 0.0)

if __name__ == '__main__':
    unittest.main()

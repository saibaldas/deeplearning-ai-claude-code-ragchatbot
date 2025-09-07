"""Tests for configuration management"""

import pytest
import os
from unittest.mock import patch

from config import Config, config


class TestConfig:
    """Test cases for Config class"""
    
    def test_config_defaults(self):
        """Test that config has correct default values"""
        test_config = Config()
        
        # Test model settings
        assert test_config.ANTHROPIC_MODEL == "claude-sonnet-4-20250514"
        assert test_config.EMBEDDING_MODEL == "all-MiniLM-L6-v2"
        
        # Test document processing settings
        assert test_config.CHUNK_SIZE == 800
        assert test_config.CHUNK_OVERLAP == 100
        assert test_config.MAX_RESULTS == 5  # This was the critical bug we fixed
        assert test_config.MAX_HISTORY == 2
        
        # Test database paths
        assert test_config.CHROMA_PATH == "./chroma_db"
    
    def test_config_with_env_var(self):
        """Test config reads from environment variables"""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_api_key'}):
            test_config = Config()
            assert test_config.ANTHROPIC_API_KEY == 'test_api_key'
    
    def test_config_without_env_var(self):
        """Test config defaults when no environment variable set"""
        # Test with empty environment
        with patch.dict(os.environ, {}, clear=True):
            test_config = Config()
            assert test_config.ANTHROPIC_API_KEY == ""
    
    def test_global_config_instance(self):
        """Test that global config instance is properly initialized"""
        assert config is not None
        assert isinstance(config, Config)
        assert config.MAX_RESULTS == 5  # Verify our critical fix is in place
    
    def test_config_values_are_correct_types(self):
        """Test that config values have correct types"""
        test_config = Config()
        
        # String values
        assert isinstance(test_config.ANTHROPIC_API_KEY, str)
        assert isinstance(test_config.ANTHROPIC_MODEL, str)
        assert isinstance(test_config.EMBEDDING_MODEL, str)
        assert isinstance(test_config.CHROMA_PATH, str)
        
        # Integer values
        assert isinstance(test_config.CHUNK_SIZE, int)
        assert isinstance(test_config.CHUNK_OVERLAP, int)
        assert isinstance(test_config.MAX_RESULTS, int)
        assert isinstance(test_config.MAX_HISTORY, int)
        
        # Verify reasonable ranges
        assert test_config.CHUNK_SIZE > 0
        assert test_config.CHUNK_OVERLAP >= 0
        assert test_config.MAX_RESULTS > 0  # This was 0 and caused the bug
        assert test_config.MAX_HISTORY > 0
    
    def test_chunk_overlap_less_than_chunk_size(self):
        """Test that chunk overlap is less than chunk size"""
        test_config = Config()
        assert test_config.CHUNK_OVERLAP < test_config.CHUNK_SIZE
    
    def test_config_can_be_modified_for_testing(self):
        """Test that config values can be overridden for testing"""
        test_config = Config(
            CHUNK_SIZE=1000,
            MAX_RESULTS=10,
            CHROMA_PATH="./test_db"
        )
        
        assert test_config.CHUNK_SIZE == 1000
        assert test_config.MAX_RESULTS == 10
        assert test_config.CHROMA_PATH == "./test_db"
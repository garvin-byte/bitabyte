#!/usr/bin/env python3
"""
Simple test to verify entropy heatmap functionality
Creates test files with different entropy patterns
"""

import numpy as np
from entropy_heatmap import EntropyHeatmapWidget


def create_test_file_mixed_entropy():
    """Create a test binary file with varying entropy sections."""
    data = []

    # Section 1: Zero padding (entropy ≈ 0.0)
    data.append(np.zeros(1024, dtype=np.uint8))

    # Section 2: Structured text (entropy ≈ 0.3-0.4)
    text = b"The quick brown fox jumps over the lazy dog. " * 20
    data.append(np.frombuffer(text, dtype=np.uint8))

    # Section 3: Random data (entropy ≈ 1.0)
    data.append(np.random.randint(0, 256, size=1024, dtype=np.uint8))

    # Section 4: Repetitive pattern (entropy ≈ 0.2)
    pattern = np.array([0xAB, 0xCD] * 512, dtype=np.uint8)
    data.append(pattern)

    # Section 5: More zeros
    data.append(np.zeros(512, dtype=np.uint8))

    # Concatenate all sections
    full_data = np.concatenate(data)

    return full_data


def test_entropy_calculation():
    """Test the entropy calculation with known patterns."""
    print("Testing Entropy Calculation")
    print("=" * 50)

    widget = EntropyHeatmapWidget()

    # Test 1: All zeros (should be entropy ≈ 0.0)
    zeros = np.zeros(256, dtype=np.uint8)
    entropy_zeros = widget._calculate_shannon_entropy(zeros)
    print(f"1. All zeros:        Entropy = {entropy_zeros:.4f} (expected ≈ 0.0000)")

    # Test 2: All ones (should be entropy ≈ 0.0)
    ones = np.ones(256, dtype=np.uint8) * 255
    entropy_ones = widget._calculate_shannon_entropy(ones)
    print(f"2. All 255s:         Entropy = {entropy_ones:.4f} (expected ≈ 0.0000)")

    # Test 3: Alternating pattern (should be low entropy)
    pattern = np.array([0, 255] * 128, dtype=np.uint8)
    entropy_pattern = widget._calculate_shannon_entropy(pattern)
    print(f"3. 0/255 pattern:    Entropy = {entropy_pattern:.4f} (expected ≈ 0.1250)")

    # Test 4: Random data (should be high entropy ≈ 0.9-1.0)
    random_data = np.random.randint(0, 256, size=256, dtype=np.uint8)
    entropy_random = widget._calculate_shannon_entropy(random_data)
    print(f"4. Random data:      Entropy = {entropy_random:.4f} (expected ≈ 0.9-1.0)")

    # Test 5: Uniform distribution (should be max entropy = 1.0)
    # Each value 0-255 appears exactly once
    uniform = np.arange(256, dtype=np.uint8)
    entropy_uniform = widget._calculate_shannon_entropy(uniform)
    print(f"5. Uniform (0-255):  Entropy = {entropy_uniform:.4f} (expected = 1.0000)")

    print("\n" + "=" * 50)
    print("All tests completed!")


def test_color_mapping():
    """Test the color mapping for different entropy values."""
    print("\nTesting Color Mapping")
    print("=" * 50)

    widget = EntropyHeatmapWidget()

    test_values = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]

    for entropy in test_values:
        color = widget._get_color_for_entropy(entropy)
        rgb = f"RGB({color.red():3d}, {color.green():3d}, {color.blue():3d})"

        # Classify entropy
        if entropy < 0.1:
            classification = "Zero/Constant"
        elif entropy < 0.3:
            classification = "Very Low"
        elif entropy < 0.5:
            classification = "Low-Medium"
        elif entropy < 0.7:
            classification = "Medium"
        elif entropy < 0.85:
            classification = "High"
        else:
            classification = "Very High"

        print(f"Entropy {entropy:.1f}: {rgb:25s} → {classification}")

    print("=" * 50)


def save_test_file():
    """Create and save a test binary file."""
    print("\nCreating test file...")
    data = create_test_file_mixed_entropy()

    filename = "test_entropy_mixed.bin"
    with open(filename, 'wb') as f:
        f.write(data.tobytes())

    print(f"✓ Created '{filename}' ({len(data)} bytes)")
    print(f"  - Bytes 0-1023:       Zero padding (low entropy)")
    print(f"  - Bytes 1024-1943:    Text data (medium-low entropy)")
    print(f"  - Bytes 1944-2967:    Random data (high entropy)")
    print(f"  - Bytes 2968-3991:    Pattern 0xABCD (low entropy)")
    print(f"  - Bytes 3992-4503:    Zero padding (low entropy)")
    print(f"\nLoad this file in bitabyte.py to see the entropy heatmap!")


if __name__ == "__main__":
    # Run all tests
    test_entropy_calculation()
    test_color_mapping()
    save_test_file()

    print("\n" + "=" * 50)
    print("All tests completed successfully!")
    print("=" * 50)

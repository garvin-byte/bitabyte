#!/usr/bin/env python3
"""
Complete Bit Processing Command Line Tool - Latest Version

Features:
- Multi-bit delta encoding (delta 1, 2, 3, etc.)
- Complex takeskip patterns with invert/reverse (t1s3i4r8)
- Block interleaving and deinterleaving with proper parameters
- Sync pattern detection with occurrence selection, skip option, and error tolerance
- External command execution for custom processing
- All operations execute in command-line order

Usage examples:
    python bitprocess.py -file input.bin -delta1 -takeskip t2s3 -delta1 -output output.bin
    python bitprocess.py -file data.bin -sync 0x1ACFFC1D -delta 2 -takeskip t8s2 -output result.bin
    python bitprocess.py -file signal.bin -deinterleave c8r16s1 -takeskip t1s3i4r8 -output processed.bin
    python bitprocess.py -file dvb.bin -sync 0x47 -occurrence 0 -skip -deinterleave c12r17s1 -takeskip t188s0 -output transport.bin
    python bitprocess.py -file new.bin -sync 00111111 -error 15 -output processed_output.bin
    python bitprocess.py -file input.bin -delta1 -external "custom_decoder.exe {input} {output}" -output final.bin
"""

import argparse
import numpy as np
import sys
import re
import subprocess
import os
from typing import List, Tuple, Union


class BitProcessorCLI:
    def __init__(self):
        self.operations = []
        self.data = None

    def load_file(self, filename: str) -> np.ndarray:
        """Load binary file as bit array"""
        try:
            with open(filename, 'rb') as f:
                byte_data = np.frombuffer(f.read(), dtype=np.uint8)
            return np.unpackbits(byte_data)
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading file '{filename}': {e}")
            sys.exit(1)

    def save_file(self, bits: np.ndarray, filename: str):
        """Save bit array to binary file"""
        try:
            # Pad to byte boundary
            padding = (8 - (len(bits) % 8)) % 8
            if padding:
                bits = np.append(bits, np.zeros(padding, dtype=np.uint8))

            byte_data = np.packbits(bits)
            with open(filename, 'wb') as f:
                f.write(byte_data.tobytes())
        except Exception as e:
            print(f"Error writing file '{filename}': {e}")
            sys.exit(1)

    def hex_to_bits(self, hex_string: str) -> np.ndarray:
        """Convert hex string to bit array"""
        if hex_string.startswith('0x') or hex_string.startswith('0X'):
            hex_string = hex_string[2:]

        try:
            value = int(hex_string, 16)
            bit_length = len(hex_string) * 4
            binary_str = format(value, f'0{bit_length}b')
            return np.array([int(b) for b in binary_str], dtype=np.uint8)
        except ValueError:
            print(f"Error: Invalid hex pattern '{hex_string}'")
            sys.exit(1)

    def binary_to_bits(self, binary_string: str) -> np.ndarray:
        """Convert binary string to bit array"""
        try:
            return np.array([int(b) for b in binary_string], dtype=np.uint8)
        except ValueError:
            print(f"Error: Invalid binary pattern '{binary_string}'")
            sys.exit(1)

    def find_sync_pattern(self, data: np.ndarray, pattern: np.ndarray, error_percent: int = 0) -> List[int]:
        """Find all sync pattern positions with optional error tolerance"""
        positions = []
        pattern_len = len(pattern)

        # Calculate maximum allowed bit errors
        max_errors = int((error_percent / 100.0) * pattern_len)

        for i in range(len(data) - pattern_len + 1):
            segment = data[i:i + pattern_len]

            if max_errors == 0:
                # Exact match
                if np.array_equal(segment, pattern):
                    positions.append(i)
            else:
                # Count bit differences
                differences = np.sum(segment != pattern)
                if differences <= max_errors:
                    positions.append(i)

        return positions

    def apply_sync(self, data: np.ndarray, pattern_str: str, occurrence: int = 0,
                   skip_pattern: bool = False, error_percent: int = 0) -> np.ndarray:
        """Apply sync operation with optional skip and error tolerance"""
        # Determine if hex or binary
        if pattern_str.startswith('0x') or pattern_str.startswith('0X'):
            pattern = self.hex_to_bits(pattern_str)
        else:
            pattern = self.binary_to_bits(pattern_str)

        positions = self.find_sync_pattern(data, pattern, error_percent)

        if len(positions) == 0:
            if error_percent > 0:
                print(f"Error: Sync pattern '{pattern_str}' not found (even with ±{error_percent}% tolerance)")
            else:
                print(f"Error: Sync pattern '{pattern_str}' not found")
            sys.exit(1)

        if occurrence >= len(positions):
            print(f"Error: Occurrence {occurrence} not found. Only {len(positions)} occurrences exist.")
            sys.exit(1)

        sync_pos = positions[occurrence]

        if skip_pattern:
            # Skip the sync pattern - start after it
            sync_pos += len(pattern)
            if error_percent > 0:
                print(
                    f"Sync pattern found at bit position {positions[occurrence]} (±{error_percent}% tolerance), starting after pattern at {sync_pos}")
            else:
                print(
                    f"Sync pattern found at bit position {positions[occurrence]}, starting after pattern at {sync_pos}")
        else:
            if error_percent > 0:
                print(f"Sync pattern found at bit position {sync_pos} (±{error_percent}% tolerance)")
            else:
                print(f"Sync pattern found at bit position {sync_pos}")

        return data[sync_pos:]

    def apply_delta(self, data: np.ndarray, window: int = 1) -> np.ndarray:
        """Apply delta encoding with specified window size"""
        if len(data) == 0:
            return data

        if window <= 0:
            print(f"Error: Delta window must be > 0, got {window}")
            sys.exit(1)

        if window == 1:
            # Classic delta encoding - optimized version
            result = np.zeros_like(data)
            result[0] = data[0]
            result[1:] = np.bitwise_xor(data[1:], data[:-1])
            return result

        # Multi-bit delta encoding
        result = np.zeros_like(data)
        result[:window] = data[:window]

        for i in range(window, len(data), window):
            end_pos = min(i + window, len(data))
            current_bits = data[i:end_pos]
            previous_bits = data[i - window:i - window + len(current_bits)]
            result[i:end_pos] = np.bitwise_xor(current_bits, previous_bits)

        return result

    def parse_takeskip(self, takeskip_str: str) -> dict:
        """Parse takeskip format: t2s3 -> {'take': 2, 'skip': 3} or t1s3i4r8 -> complex pattern"""
        takeskip_str_lower = takeskip_str.lower()

        try:
            i = 0
            pattern = {}
            current_op = None
            current_num = ""
            operations_order = []  # Track order of t and s operations

            while i <= len(takeskip_str_lower):
                if i == len(takeskip_str_lower):
                    if current_op and current_num:
                        pattern[current_op] = int(current_num)
                    break

                char = takeskip_str_lower[i]

                if char in 'tsir':
                    if current_op and current_num:
                        pattern[current_op] = int(current_num)
                        current_num = ""
                    current_op = char
                    if char in 'ts':
                        operations_order.append(char)
                elif char.isdigit():
                    current_num += char
                else:
                    raise ValueError(f"Invalid character '{char}'")

                i += 1

            if 't' not in pattern:
                pattern['t'] = 1
            if 's' not in pattern:
                pattern['s'] = 0

            # Determine order based on which came first
            if operations_order:
                if operations_order[0] == 's':
                    pattern['order'] = 'st'  # Skip first
                else:
                    pattern['order'] = 'ts'  # Take first
            else:
                pattern['order'] = 'ts'  # Default

            if pattern['t'] < 0:
                raise ValueError("Take count must be >= 0")
            if pattern['s'] < 0:
                raise ValueError("Skip count must be >= 0")
            if 'i' in pattern and pattern['i'] <= 0:
                raise ValueError("Invert count must be > 0")
            if 'r' in pattern and pattern['r'] <= 0:
                raise ValueError("Reverse count must be > 0")

            return pattern

        except ValueError as e:
            print(f"Error: Invalid takeskip format '{takeskip_str}'. {e}")
            sys.exit(1)

    def apply_takeskip(self, data: np.ndarray, pattern: dict) -> np.ndarray:
        """Apply complex take/skip pattern with optional invert and reverse"""
        take = pattern['t']
        skip = pattern['s']
        invert_count = pattern.get('i', 0)
        reverse_count = pattern.get('r', 0)
        order = pattern.get('order', 'ts')  # Get the order, default to take-skip

        if (take == 1 or take == 0) and skip == 0 and (invert_count > 0 or reverse_count > 0):
            result = data.copy()

            if invert_count > 0:
                temp = []
                for i in range(0, len(result), invert_count):
                    end_idx = min(i + invert_count, len(result))
                    chunk = result[i:end_idx]
                    temp.extend(1 - chunk)
                result = np.array(temp, dtype=np.uint8)

            if reverse_count > 0:
                temp = []
                for i in range(0, len(result), reverse_count):
                    end_idx = min(i + reverse_count, len(result))
                    chunk = result[i:end_idx]
                    temp.extend(chunk[::-1])
                result = np.array(temp, dtype=np.uint8)

            return result

        result = []
        position = 0

        # Determine which operation comes first based on order
        if order == 'st':  # Skip first, then take
            while position < len(data):
                # Skip first
                position += skip
                if position >= len(data):
                    break

                # Then take
                take_end = min(position + take, len(data))
                taken_bits = data[position:take_end]

                if len(taken_bits) > 0:
                    if invert_count > 0:
                        invert_end = min(invert_count, len(taken_bits))
                        taken_bits = taken_bits.copy()
                        taken_bits[:invert_end] = 1 - taken_bits[:invert_end]

                    if reverse_count > 0:
                        if reverse_count >= len(taken_bits):
                            taken_bits = taken_bits[::-1]
                        else:
                            taken_bits = taken_bits.copy()
                            for i in range(0, len(taken_bits), reverse_count):
                                end_idx = min(i + reverse_count, len(taken_bits))
                                taken_bits[i:end_idx] = taken_bits[i:end_idx][::-1]

                    result.extend(taken_bits)

                position = take_end
        else:  # Take first, then skip (default 'ts' order)
            while position < len(data):
                take_end = min(position + take, len(data))
                taken_bits = data[position:take_end]

                if len(taken_bits) > 0:
                    if invert_count > 0:
                        invert_end = min(invert_count, len(taken_bits))
                        taken_bits = taken_bits.copy()
                        taken_bits[:invert_end] = 1 - taken_bits[:invert_end]

                    if reverse_count > 0:
                        if reverse_count >= len(taken_bits):
                            taken_bits = taken_bits[::-1]
                        else:
                            taken_bits = taken_bits.copy()
                            for i in range(0, len(taken_bits), reverse_count):
                                end_idx = min(i + reverse_count, len(taken_bits))
                                taken_bits[i:end_idx] = taken_bits[i:end_idx][::-1]

                    result.extend(taken_bits)

                position = min(position + take + skip, len(data))

        return np.array(result, dtype=np.uint8)

    def apply_invert(self, data: np.ndarray, count: int = None) -> np.ndarray:
        """Apply bit inversion"""
        if count is None:
            return 1 - data
        else:
            result = data.copy()
            result[:min(count, len(data))] = 1 - result[:min(count, len(data))]
            return result

    def apply_xor(self, data: np.ndarray, pattern_str: str) -> np.ndarray:
        """Apply XOR with pattern"""
        if pattern_str.startswith('0x') or pattern_str.startswith('0X'):
            pattern = self.hex_to_bits(pattern_str)
        else:
            pattern = self.binary_to_bits(pattern_str)

        if len(pattern) == 0:
            return data

        n_repeats = (len(data) + len(pattern) - 1) // len(pattern)
        extended_pattern = np.tile(pattern, n_repeats)[:len(data)]

        return np.bitwise_xor(data, extended_pattern)

    def apply_LRS(self, data: np.ndarray, taps_str: str) -> np.ndarray:
        """Apply feedthrough LRS descrambler"""
        taps_str = taps_str.replace(',', ' ')
        try:
            taps = [int(t.strip()) for t in taps_str.split() if t.strip()]
        except ValueError:
            print(f"Error: Invalid LRS tap format '{taps_str}'")
            sys.exit(1)

        if len(taps) < 2:
            print(f"Error: LRS needs at least 2 tap points")
            sys.exit(1)

        if any(t < 0 for t in taps):
            print(f"Error: LRS tap points must be >= 0")
            sys.exit(1)

        max_tap = max(taps)
        if max_tap >= len(data):
            print(f"Error: LRS tap point {max_tap} exceeds data length {len(data)}")
            sys.exit(1)

        result = data.copy()

        for i in range(max_tap, len(result)):
            xor_result = 0
            for tap in taps:
                xor_result ^= data[i - max_tap + tap]
            result[i] = data[i] ^ xor_result

        return result

    def apply_LRS_additive(self, data: np.ndarray, taps_str: str, fill_str: str, length: int) -> np.ndarray:
        """Apply additive LRS"""
        taps_str = taps_str.replace(',', ' ')
        try:
            taps = [int(t.strip()) for t in taps_str.split() if t.strip()]
        except ValueError:
            print(f"Error: Invalid LRS tap format '{taps_str}'")
            sys.exit(1)

        if len(taps) < 2:
            print(f"Error: LRS needs at least 2 tap points")
            sys.exit(1)

        if any(t < 0 for t in taps):
            print(f"Error: LRS tap points must be >= 0")
            sys.exit(1)

        if fill_str.startswith('0x') or fill_str.startswith('0X'):
            fill_bits = self.hex_to_bits(fill_str)
        else:
            fill_bits = self.binary_to_bits(fill_str)

        if fill_bits is None or len(fill_bits) == 0:
            print(f"Error: Invalid initial fill format '{fill_str}'")
            sys.exit(1)

        max_tap = max(taps)
        register_size = max_tap + 1

        if np.all(fill_bits == 0):
            print(f"Error: Initial fill cannot be all zeros")
            sys.exit(1)

        if len(fill_bits) < register_size:
            fill_bits = np.concatenate([np.zeros(register_size - len(fill_bits), dtype=np.uint8), fill_bits])
        elif len(fill_bits) > register_size:
            fill_bits = fill_bits[-register_size:]

        if length <= 0:
            print(f"Error: Output length must be > 0, got {length}")
            sys.exit(1)

        register = fill_bits.copy()
        LRS_sequence = []

        for _ in range(length):
            LRS_sequence.append(register[0])
            feedback = 0
            for tap in taps:
                feedback ^= register[tap]
            register = np.roll(register, -1)
            register[-1] = feedback

        LRS_sequence = np.array(LRS_sequence, dtype=np.uint8)

        if len(LRS_sequence) < len(data):
            n_repeats = (len(data) + len(LRS_sequence) - 1) // len(LRS_sequence)
            extended_sequence = np.tile(LRS_sequence, n_repeats)[:len(data)]
            return np.bitwise_xor(data, extended_sequence)
        else:
            min_len = min(len(data), len(LRS_sequence))
            return np.bitwise_xor(data[:min_len], LRS_sequence[:min_len])

    def apply_reverse(self, data: np.ndarray, block_size: int) -> np.ndarray:
        """Apply block reversal"""
        n_blocks = len(data) // block_size
        remainder = len(data) % block_size

        if n_blocks == 0:
            return data[::-1] if len(data) <= block_size else data

        main_bits = data[:n_blocks * block_size].reshape(n_blocks, block_size)
        reversed_blocks = np.flip(main_bits, axis=1)
        result = reversed_blocks.flatten()

        if remainder > 0:
            remainder_bits = data[-remainder:]
            result = np.concatenate([result, remainder_bits[::-1]])

        return result

    def apply_block_interleaver(self, data: np.ndarray, block_size: int) -> np.ndarray:
        """Apply block interleaver"""
        if block_size <= 0:
            print(f"Error: Block size must be > 0, got {block_size}")
            sys.exit(1)

        if len(data) == 0:
            return data

        n_blocks = len(data) // block_size
        remainder = len(data) % block_size

        if n_blocks == 0:
            return data

        main_bits = data[:n_blocks * block_size].reshape(n_blocks, block_size)
        interleaved = main_bits.T.flatten()

        if remainder > 0:
            interleaved = np.concatenate([interleaved, data[-remainder:]])

        return interleaved

    def apply_block_deinterleaver(self, data: np.ndarray, cols: int, rows: int, symbol_size: int = 1) -> np.ndarray:
        """Apply block deinterleaver

        Standard block deinterleaving:
        - Input data is written row-by-row into a (rows x cols) matrix
        - Output is read column-by-column from the matrix
        """
        if cols <= 0 or rows <= 0 or symbol_size <= 0:
            print(f"Error: Columns, rows, and symbol size must be > 0")
            sys.exit(1)

        if len(data) == 0:
            return data

        matrix_size = cols * rows * symbol_size
        result = []

        for start_idx in range(0, len(data), matrix_size):
            chunk = data[start_idx:start_idx + matrix_size]

            if len(chunk) < matrix_size:
                result.extend(chunk)
                break

            if symbol_size == 1:
                # Reshape as (rows, cols) - data fills row-by-row
                matrix = chunk.reshape(rows, cols)
                # Transpose to read column-by-column
                deinterleaved = matrix.T.flatten()
            else:
                # Symbol-wise deinterleaving
                n_symbols = len(chunk) // symbol_size
                symbols = chunk.reshape(n_symbols, symbol_size)
                # Reshape as (rows, cols, symbol_size)
                symbol_matrix = symbols.reshape(rows, cols, symbol_size)
                # Transpose (1, 0, 2) swaps rows and cols, keeps symbols intact
                deinterleaved_symbols = symbol_matrix.transpose(1, 0, 2)
                deinterleaved = deinterleaved_symbols.flatten()

            result.extend(deinterleaved)

        return np.array(result, dtype=np.uint8)

    def apply_bitmap(self, data: np.ndarray, size: int, mapping: List[int]) -> np.ndarray:
        """Apply bit mapping to rearrange bits within groups"""
        if size not in [4, 8, 16]:
            print(f"Error: Bit mapping size must be 4, 8, or 16, got {size}")
            sys.exit(1)

        if len(mapping) != size:
            print(f"Error: Mapping must have exactly {size} elements, got {len(mapping)}")
            sys.exit(1)

        if set(mapping) != set(range(size)):
            print(f"Error: Mapping must contain each value from 0 to {size - 1} exactly once")
            sys.exit(1)

        if len(data) == 0:
            return data

        result = []
        for i in range(0, len(data), size):
            group = data[i:i + size]

            if len(group) < size:
                # Pad incomplete group with zeros
                group = np.append(group, np.zeros(size - len(group), dtype=np.uint8))

            # Remap bits according to mapping
            remapped = np.zeros(size, dtype=np.uint8)
            for out_pos, in_pos in enumerate(mapping):
                remapped[out_pos] = group[in_pos]

            result.extend(remapped)

        return np.array(result, dtype=np.uint8)

    def parse_deinterleave_params(self, param_str: str) -> tuple:
        """Parse deinterleave parameters"""
        param_str = param_str.lower()

        if not param_str.startswith('c') or 'r' not in param_str:
            print(f"Error: Invalid deinterleave format '{param_str}'")
            sys.exit(1)

        try:
            match = re.match(r'c(\d+)r(\d+)(?:s(\d+))?', param_str)

            if not match:
                raise ValueError("Pattern doesn't match expected format")

            cols = int(match.group(1))
            rows = int(match.group(2))
            symbol_size = int(match.group(3)) if match.group(3) else 1

            if cols <= 0 or rows <= 0 or symbol_size <= 0:
                raise ValueError("All parameters must be > 0")

            return cols, rows, symbol_size

        except (ValueError, AttributeError) as e:
            print(f"Error: Invalid deinterleave format '{param_str}'. {e}")
            sys.exit(1)

    def apply_external(self, data: np.ndarray, command: str, temp_input: str = "temp_external_input.bin",
                       temp_output: str = "temp_external_output.bin") -> np.ndarray:
        """Execute external command and load its output

        Args:
            data: Input bit array
            command: Command to execute (use {input} and {output} as placeholders)
            temp_input: Temporary input filename
            temp_output: Temporary output filename
        """
        # Save current data to temp file
        self.save_file(data, temp_input)

        # Replace placeholders in command
        cmd = command.replace('{input}', temp_input).replace('{output}', temp_output)

        print(f"  Executing external command: {cmd}")

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode != 0:
                print(f"Error: External command failed with return code {result.returncode}")
                if result.stderr:
                    print(f"STDERR: {result.stderr}")
                # Clean up input file
                if os.path.exists(temp_input):
                    os.remove(temp_input)
                sys.exit(1)

            if not os.path.exists(temp_output):
                print(f"Error: External command did not create output file '{temp_output}'")
                # Clean up input file
                if os.path.exists(temp_input):
                    os.remove(temp_input)
                sys.exit(1)

            # Load the output
            output_data = self.load_file(temp_output)

            # Clean up temp files
            if os.path.exists(temp_input):
                os.remove(temp_input)
            if os.path.exists(temp_output):
                os.remove(temp_output)

            print(f"  External command completed successfully")
            return output_data

        except subprocess.TimeoutExpired:
            print(f"Error: External command timed out after 5 minutes")
            # Clean up temp files
            if os.path.exists(temp_input):
                os.remove(temp_input)
            if os.path.exists(temp_output):
                os.remove(temp_output)
            sys.exit(1)
        except Exception as e:
            print(f"Error executing external command: {e}")
            # Clean up temp files
            if os.path.exists(temp_input):
                os.remove(temp_input)
            if os.path.exists(temp_output):
                os.remove(temp_output)
            sys.exit(1)

    def process_operations(self, data: np.ndarray, operations: List[Tuple[str, str]]) -> np.ndarray:
        """Process operations in order"""
        current_data = data

        for i, (op_type, op_value) in enumerate(operations):
            print(f"Step {i + 1}: Applying {op_type} {op_value if op_value else ''}")
            print(f"  Input length: {len(current_data)} bits")

            if op_type == 'sync':
                occurrence = getattr(self, '_sync_occurrence', 0)
                skip_pattern = getattr(self, '_sync_skip', False)
                error_percent = getattr(self, '_sync_error', 0)
                current_data = self.apply_sync(current_data, op_value, occurrence, skip_pattern, error_percent)

            elif op_type == 'zeropad_frame':
                # Parse pattern
                if op_value.startswith('0x') or op_value.startswith('0X'):
                    pattern = self.hex_to_bits(op_value)
                else:
                    pattern = self.binary_to_bits(op_value)
                # Use existing sync error tolerance if set
                error_percent = getattr(self, '_sync_error', 0)
                positions = self.find_sync_pattern(current_data, pattern, error_percent)
                if not positions:
                    if error_percent > 0:
                        print(
                            f"Error: Frame sync pattern '{op_value}' not found (even with ±{error_percent}% tolerance)")
                    else:
                        print(f"Error: Frame sync pattern '{op_value}' not found")
                    sys.exit(1)
                # Extract frames
                frames = []
                pattern_len = len(pattern)
                for j, pos in enumerate(positions):
                    start_pos = pos  # Start at the pattern itself
                    if j + 1 < len(positions):
                        end_pos = positions[j + 1]
                    else:
                        end_pos = len(current_data)
                    frame = current_data[start_pos:end_pos]
                    # Only add frames that have meaningful length
                    if len(frame) >= pattern_len:
                        frames.append(frame)
                if not frames:
                    print(f"Error: No valid frames found")
                    sys.exit(1)
                max_len = max(len(frame) for frame in frames)
                padded_frames = []
                for frame in frames:
                    if len(frame) < max_len:
                        padding = np.zeros(max_len - len(frame), dtype=np.uint8)
                        padded_frame = np.concatenate([frame, padding])
                    else:
                        padded_frame = frame
                    padded_frames.append(padded_frame)
                current_data = np.concatenate(padded_frames)
                print(f"  Extracted {len(frames)} frames, padded to {max_len} bits each")
            elif op_type == 'delta':
                window = int(op_value) if op_value else 1
                current_data = self.apply_delta(current_data, window)

            elif op_type == 'takeskip':
                pattern = self.parse_takeskip(op_value)
                current_data = self.apply_takeskip(current_data, pattern)

            elif op_type == 'invert':
                if op_value:
                    try:
                        count = int(op_value)
                        current_data = self.apply_invert(current_data, count)
                    except ValueError:
                        print(f"Error: Invalid invert count '{op_value}'")
                        sys.exit(1)
                else:
                    current_data = self.apply_invert(current_data)

            elif op_type == 'xor':
                current_data = self.apply_xor(current_data, op_value)

            elif op_type == 'LRS':
                current_data = self.apply_LRS(current_data, op_value)

            elif op_type == 'LRS_add':
                parts = op_value.split()
                if len(parts) < 3:
                    print(f"Error: LRS_add requires taps, fill, and length")
                    sys.exit(1)
                taps = parts[0]
                fill = parts[1]
                length = int(parts[2])
                current_data = self.apply_LRS_additive(current_data, taps, fill, length)

            elif op_type == 'reverse':
                try:
                    block_size = int(op_value)
                    current_data = self.apply_reverse(current_data, block_size)
                except ValueError:
                    print(f"Error: Invalid reverse block size '{op_value}'")
                    sys.exit(1)

            elif op_type == 'interleave':
                try:
                    block_size = int(op_value)
                    current_data = self.apply_block_interleaver(current_data, block_size)
                except ValueError:
                    print(f"Error: Invalid interleave block size '{op_value}'")
                    sys.exit(1)

            elif op_type == 'deinterleave':
                try:
                    cols, rows, symbol_size = self.parse_deinterleave_params(op_value)
                    current_data = self.apply_block_deinterleaver(current_data, cols, rows, symbol_size)
                except ValueError:
                    print(f"Error: Invalid deinterleave parameters '{op_value}'")
                    sys.exit(1)

            elif op_type == 'external':
                current_data = self.apply_external(current_data, op_value)

            elif op_type == 'bitmap':
                try:
                    parts = op_value.split()
                    size = int(parts[0])
                    mapping_str = parts[1] if len(parts) > 1 else parts[0]
                    mapping = [int(x.strip()) for x in mapping_str.split(',')]
                    current_data = self.apply_bitmap(current_data, size, mapping)
                except (ValueError, IndexError) as e:
                    print(f"Error: Invalid bitmap parameters '{op_value}': {e}")
                    sys.exit(1)
            elif op_type == 'zeropad_frame_startstop':
                # Parse start and stop patterns
                parts = op_value.split()
                if len(parts) < 2:
                    print(f"Error: zeropad_frame_startstop requires start and stop patterns")
                    sys.exit(1)

                start_pattern_str = parts[0]
                stop_pattern_str = parts[1]

                if start_pattern_str.startswith('0x') or start_pattern_str.startswith('0X'):
                    start_pattern = self.hex_to_bits(start_pattern_str)
                else:
                    start_pattern = self.binary_to_bits(start_pattern_str)

                if stop_pattern_str.startswith('0x') or stop_pattern_str.startswith('0X'):
                    stop_pattern = self.hex_to_bits(stop_pattern_str)
                else:
                    stop_pattern = self.binary_to_bits(stop_pattern_str)

                # Use existing sync error tolerance if set
                error_percent = getattr(self, '_sync_error', 0)
                start_positions = self.find_sync_pattern(current_data, start_pattern, error_percent)
                stop_positions = self.find_sync_pattern(current_data, stop_pattern, error_percent)

                if not start_positions:
                    if error_percent > 0:
                        print(
                            f"Error: Start pattern '{start_pattern_str}' not found (even with ±{error_percent}% tolerance)")
                    else:
                        print(f"Error: Start pattern '{start_pattern_str}' not found")
                    sys.exit(1)

                if not stop_positions:
                    if error_percent > 0:
                        print(
                            f"Error: Stop pattern '{stop_pattern_str}' not found (even with ±{error_percent}% tolerance)")
                    else:
                        print(f"Error: Stop pattern '{stop_pattern_str}' not found")
                    sys.exit(1)

                # Extract frames from start to stop
                frames = []
                start_len = len(start_pattern)
                stop_len = len(stop_pattern)

                for start_pos in start_positions:
                    # Find the first stop position after this start
                    matching_stops = [s for s in stop_positions if s > start_pos]

                    if matching_stops:
                        stop_pos = matching_stops[0]
                        # Include from start pattern through end of stop pattern
                        frame = current_data[start_pos:stop_pos + stop_len]
                        frames.append(frame)

                if not frames:
                    print(f"Error: No valid start/stop frame pairs found")
                    sys.exit(1)

                max_len = max(len(frame) for frame in frames)
                padded_frames = []
                for frame in frames:
                    if len(frame) < max_len:
                        padding = np.zeros(max_len - len(frame), dtype=np.uint8)
                        padded_frame = np.concatenate([frame, padding])
                    else:
                        padded_frame = frame
                    padded_frames.append(padded_frame)

                current_data = np.concatenate(padded_frames)
                print(f"  Extracted {len(frames)} frames (start→stop), padded to {max_len} bits each")

            print(f"  Output length: {len(current_data)} bits")

        return current_data


def main():
    parser = argparse.ArgumentParser(
        description='Complete Bit Processing Tool - Process binary files with various bit operations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -file input.bin -delta1 -takeskip t2s3 -delta1 -output output.bin
  %(prog)s -file data.bin -sync 0x1ACFFC1D -delta 2 -takeskip t8s2 -output result.bin
  %(prog)s -file signal.bin -deinterleave c8r16s1 -takeskip t1s3i4r8 -output processed.bin
  %(prog)s -file dvb.bin -sync 0x47 -occurrence 0 -skip -deinterleave c12r17s1 -takeskip t188s0 -output transport.bin
  %(prog)s -file new.bin -sync 00111111 -error 15 -output processed_output.bin
  %(prog)s -file input.bin -delta1 -external "custom_decoder.exe {input} {output}" -output final.bin
        """
    )

    # Required arguments
    parser.add_argument('-file', required=True, help='Input binary file')

    # Operations (order matters)
    parser.add_argument('-sync', metavar='PATTERN', help='Sync to pattern (hex: 0x1234 or binary: 10110010)')
    parser.add_argument('-zeropad_frame', metavar='PATTERN',
                        help='Zero-pad frame sync: extract frames at pattern, pad to longest (hex: 0x1234 or binary: 10110010)')
    parser.add_argument('-occurrence', type=int, default=0, metavar='N',
                        help='Which sync occurrence to use (default: 0)')
    parser.add_argument('-skip', action='store_true', help='Skip sync pattern (start after pattern)')
    parser.add_argument('-error', type=int, default=0, metavar='PERCENT',
                        help='Error tolerance percentage for sync pattern (0-100, default: 0)')
    parser.add_argument('-delta1', action='store_true', help='Apply delta encoding (XOR with previous 1 bit)')
    parser.add_argument('-delta', type=int, metavar='N',
                        help='Apply delta encoding with window N (XOR with previous N bits)')
    parser.add_argument('-takeskip', metavar='PATTERN',
                        help='Complex take/skip pattern (e.g., t2s3, t1s3i4, t1s3r8, t1s3i4r8)')
    parser.add_argument('-invert', nargs='?', const='', metavar='COUNT', help='Invert bits (all if no count given)')
    parser.add_argument('-xor', metavar='PATTERN', help='XOR with pattern (hex: 0x1234 or binary: 10110010)')
    parser.add_argument('-LRS', metavar='TAPS', help='LRS descrambler with tap points (e.g., "0,1,7" or "0 1 7")')
    parser.add_argument('-LRS_add', nargs=3, metavar=('TAPS', 'FILL', 'LENGTH'),
                        help='Additive LRS: taps, initial_fill, length (e.g., "0,1,7" 11111111 255)')
    parser.add_argument('-reverse', type=int, metavar='BLOCKSIZE', help='Reverse bits in blocks of size N')
    parser.add_argument('-interleave', type=int, metavar='BLOCKSIZE', help='Block interleave with block size N')
    parser.add_argument('-deinterleave', metavar='PARAMS',
                        help='Block deinterleave: c<cols>r<rows>s<symbol_size> (e.g., c8r16s1)')
    parser.add_argument('-external', metavar='COMMAND',
                        help='Run external command (use {input} and {output} as placeholders)')
    parser.add_argument('-bitmap', nargs=2, metavar=('SIZE', 'MAPPING'),
                        help='Bit mapping: size (4/8/16) and comma-separated mapping (e.g., 4 "0,2,1,3")')

    # Output
    parser.add_argument('-output', help='Output binary file (default: stdout info only)')
    parser.add_argument('-stats', action='store_true', help='Show detailed statistics')
    parser.add_argument('-zeropad_frame_startstop', nargs=2, metavar=('START_PATTERN', 'STOP_PATTERN'),
                        help='Zero-pad frame sync with start/stop patterns: extract frames from start to stop pattern (hex: 0x1234 or binary: 10110010)')

    args = parser.parse_args()

    # Create processor
    processor = BitProcessorCLI()

    # Load input file
    print(f"Loading file: {args.file}")
    data = processor.load_file(args.file)
    print(f"Loaded {len(data)} bits ({len(data) / 8:.1f} bytes)")

    # Build operations list in order they were specified
    argv_order = []
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '-sync' and i + 1 < len(sys.argv):
            argv_order.append(('sync', sys.argv[i + 1]))
            i += 2
        elif arg == '-zeropad_frame' and i + 1 < len(sys.argv):
            argv_order.append(('zeropad_frame', sys.argv[i + 1]))
            i += 2
        elif arg == '-delta1':
            argv_order.append(('delta', '1'))
            i += 1
        elif arg == '-delta' and i + 1 < len(sys.argv):
            argv_order.append(('delta', sys.argv[i + 1]))
            i += 2
        elif arg == '-takeskip' and i + 1 < len(sys.argv):
            argv_order.append(('takeskip', sys.argv[i + 1]))
            i += 2
        elif arg == '-invert':
            if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith('-'):
                argv_order.append(('invert', sys.argv[i + 1]))
                i += 2
            else:
                argv_order.append(('invert', ''))
                i += 1
        elif arg == '-xor' and i + 1 < len(sys.argv):
            argv_order.append(('xor', sys.argv[i + 1]))
            i += 2
        elif arg == '-LRS' and i + 1 < len(sys.argv):
            argv_order.append(('LRS', sys.argv[i + 1]))
            i += 2
        elif arg == '-LRS_add' and i + 3 < len(sys.argv):
            combined = f"{sys.argv[i + 1]} {sys.argv[i + 2]} {sys.argv[i + 3]}"
            argv_order.append(('LRS_add', combined))
            i += 4
        elif arg == '-reverse' and i + 1 < len(sys.argv):
            argv_order.append(('reverse', sys.argv[i + 1]))
            i += 2
        elif arg == '-interleave' and i + 1 < len(sys.argv):
            argv_order.append(('interleave', sys.argv[i + 1]))
            i += 2
        elif arg == '-deinterleave' and i + 1 < len(sys.argv):
            argv_order.append(('deinterleave', sys.argv[i + 1]))
            i += 2
        elif arg == '-external' and i + 1 < len(sys.argv):
            argv_order.append(('external', sys.argv[i + 1]))
            i += 2
        elif arg == '-bitmap' and i + 2 < len(sys.argv):
            combined = f"{sys.argv[i + 1]} {sys.argv[i + 2]}"
            argv_order.append(('bitmap', combined))
            i += 3
        elif arg == '-zeropad_frame_startstop' and i + 2 < len(sys.argv):
            combined = f"{sys.argv[i + 1]} {sys.argv[i + 2]}"
            argv_order.append(('zeropad_frame_startstop', combined))
            i += 3
        else:
            i += 1


    # Set sync occurrence, skip flags, and error tolerance
    if args.occurrence is not None:
        processor._sync_occurrence = args.occurrence
    if args.skip:
        processor._sync_skip = True
    if args.error:
        processor._sync_error = args.error

    if not argv_order:
        print("No operations specified. Use -h for help.")
        sys.exit(1)

    # Process operations
    print(f"\nProcessing {len(argv_order)} operations in order:")
    result_data = processor.process_operations(data, argv_order)

    # Show results
    print(f"\nFinal result: {len(result_data)} bits ({len(result_data) / 8:.1f} bytes)")

    if args.stats:
        print(f"\nStatistics:")
        print(f"  Original size: {len(data)} bits")
        print(f"  Final size: {len(result_data)} bits")
        print(f"  Compression ratio: {len(result_data) / len(data):.3f}")
        print(f"  Ones in original: {np.sum(data)} ({np.sum(data) / len(data) * 100:.1f}%)")
        print(f"  Ones in result: {np.sum(result_data)} ({np.sum(result_data) / len(result_data) * 100:.1f}%)")

    # Save output
    if args.output:
        print(f"\nSaving to: {args.output}")
        processor.save_file(result_data, args.output)
        print("Done!")
    else:
        print("\nNo output file specified. Use -output filename.bin to save results.")
        print(f"First 64 bits of result: {result_data[:64]}")


if __name__ == '__main__':
    main()
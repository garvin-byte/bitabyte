#!/usr/bin/env python3
"""
Interactive Frame Data Generator

Creates binary files with structured frames and sync patterns.
Asks simple y/n questions and generates the data based on your specifications.

Usage:
    python frame_generator.py
"""

import numpy as np
import sys
import os

from common.bit_utils import bits_from_binary_string, bits_from_hex_string, parse_tap_list


class FrameDataGenerator:
    def __init__(self):
        self.data = None

    def ask_yes_no(self, question: str) -> bool:
        """Ask a yes/no question"""
        while True:
            answer = input(f"{question} (y/n): ").lower().strip()
            if answer in ['y', 'yes']:
                return True
            elif answer in ['n', 'no']:
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no")

    def ask_input(self, prompt: str, default: str = None) -> str:
        """Ask for input with optional default"""
        if default:
            answer = input(f"{prompt} (default: {default}): ").strip()
            return answer if answer else default
        else:
            while True:
                answer = input(f"{prompt}: ").strip()
                if answer:
                    return answer
                print("Please enter a value")

    def ask_integer(self, prompt: str, min_val: int = None, max_val: int = None, default: int = None) -> int:
        """Ask for an integer with validation"""
        while True:
            try:
                if default is not None:
                    answer = input(f"{prompt} (default: {default}): ").strip()
                    if not answer:
                        return default
                    value = int(answer)
                else:
                    value = int(input(f"{prompt}: ").strip())

                if min_val is not None and value < min_val:
                    print(f"Value must be >= {min_val}")
                    continue
                if max_val is not None and value > max_val:
                    print(f"Value must be <= {max_val}")
                    continue

                return value
            except ValueError:
                print("Please enter a valid integer")

    def hex_to_bits(self, hex_string: str) -> np.ndarray:
        """Convert hex string to bit array"""
        bits = bits_from_hex_string(hex_string)
        if bits is None:
            print(f"Error: Invalid hex pattern '{hex_string}'")
            return None
        return bits

    def binary_to_bits(self, binary_string: str) -> np.ndarray:
        """Convert binary string to bit array"""
        bits = bits_from_binary_string(binary_string)
        if bits is None:
            print(f"Error: Invalid binary pattern '{binary_string}'")
            return None
        return bits

    def parse_taps(self, taps_str: str) -> list:
        """Parse tap string like '0,1,7' or '0 1 7' into list"""
        taps = parse_tap_list(taps_str)
        if taps is None:
            print(f"Error: Invalid tap format '{taps_str}'")
            return None
        return taps

    def apply_additive_lrs(self, data: np.ndarray, taps: list, initial_state: str = None) -> np.ndarray:
        """
        Apply additive LRS (autonomous sequence generator)
        XORs generated sequence with data
        """
        if not taps:
            return data

        max_tap = max(taps)
        reg_size = max_tap + 1

        # Parse initial state
        if initial_state:
            if initial_state.startswith('0x') or initial_state.startswith('0X'):
                init_bits = self.hex_to_bits(initial_state)
            else:
                init_bits = self.binary_to_bits(initial_state)

            if init_bits is None or len(init_bits) != reg_size:
                print(f"Invalid initial state, using all 1's")
                register = np.ones(reg_size, dtype=np.uint8)
            else:
                register = init_bits.copy()
        else:
            # Default: all 1's
            register = np.ones(reg_size, dtype=np.uint8)

        # Check for all-zeros (invalid)
        if np.all(register == 0):
            print("Warning: Initial state all zeros is invalid, using all 1's")
            register = np.ones(reg_size, dtype=np.uint8)

        # Generate LRS sequence
        lrs_sequence = []
        max_length = min(len(data), (2 ** reg_size - 1) * 3)  # Generate enough bits

        for _ in range(max_length):
            # Output leftmost bit
            lrs_sequence.append(register[0])

            # Calculate feedback
            feedback = 0
            for tap in taps:
                if tap < len(register):
                    feedback ^= register[tap]

            # Shift and insert feedback
            register = np.roll(register, -1)
            register[-1] = feedback

        lrs_sequence = np.array(lrs_sequence, dtype=np.uint8)

        # XOR with data (tile if needed)
        if len(lrs_sequence) < len(data):
            repeats = (len(data) + len(lrs_sequence) - 1) // len(lrs_sequence)
            lrs_sequence = np.tile(lrs_sequence, repeats)

        result = np.bitwise_xor(data[:len(data)], lrs_sequence[:len(data)])

        print(f"Applied additive LRS: taps={taps}, period={(2 ** reg_size) - 1}")

        return result

    def apply_feedthrough_lrs(self, data: np.ndarray, taps: list) -> np.ndarray:
        """
        Apply feedthrough LRS (descrambler mode)
        Processes data through shift register with feedback
        """
        if not taps:
            return data

        max_tap = max(taps)
        reg_size = max_tap + 1

        # Initialize register with zeros
        register = np.zeros(reg_size, dtype=np.uint8)
        result = []

        # Feed each data bit through
        for input_bit in data:
            # Calculate feedback
            feedback = 0
            for tap in taps:
                if tap < len(register):
                    feedback ^= register[tap]

            # Output is input XOR feedback
            output_bit = input_bit ^ feedback
            result.append(output_bit)

            # Shift in input bit
            register = np.roll(register, -1)
            register[-1] = input_bit

        print(f"Applied feedthrough LRS: taps={taps}, reg_size={reg_size}")

        return np.array(result, dtype=np.uint8)

    def save_file(self, bits: np.ndarray, filename: str):
        """Save bit array to binary file in current directory"""
        try:
            # Save in current directory
            filepath = os.path.join(".", filename)

            # Pad to byte boundary
            padding = (8 - (len(bits) % 8)) % 8
            if padding:
                bits = np.append(bits, np.zeros(padding, dtype=np.uint8))
                print(f"Added {padding} padding bits to align to byte boundary")

            byte_data = np.packbits(bits)
            with open(filepath, 'wb') as f:
                f.write(byte_data.tobytes())

            print(f"Saved to {filepath}")
            print(f"  File size: {len(byte_data)} bytes ({len(bits)} bits)")
            return True

        except Exception as e:
            print(f"Error writing file '{filename}': {e}")
            return False

    def generate_random_data(self, num_bits: int) -> np.ndarray:
        """Generate random bits"""
        return np.random.randint(0, 2, num_bits, dtype=np.uint8)

    def generate_pattern_data(self, num_bits: int, pattern: str) -> np.ndarray:
        """Generate data based on repeating pattern"""
        if pattern.startswith('0x') or pattern.startswith('0X'):
            pattern_bits = self.hex_to_bits(pattern)
        else:
            pattern_bits = self.binary_to_bits(pattern)

        if pattern_bits is None:
            return self.generate_random_data(num_bits)

        # Repeat pattern to fill num_bits
        repeats = (num_bits + len(pattern_bits) - 1) // len(pattern_bits)
        full_pattern = np.tile(pattern_bits, repeats)
        return full_pattern[:num_bits]

    def create_framed_data(self, sync_pattern: str, num_frames: int,
                           frame_length_mode: str = "static", static_frame_width: int = 188,
                           min_frame_width: int = 100, max_frame_width: int = 200,
                           include_counter: bool = False,
                           data_type: str = "random", data_pattern: str = None,
                           lrs_mode: str = None, lrs_taps: list = None,
                           lrs_initial_state: str = None) -> np.ndarray:
        """Create data with frames and sync patterns"""

        # Convert sync pattern to bits
        if sync_pattern.startswith('0x') or sync_pattern.startswith('0X'):
            sync_bits = self.hex_to_bits(sync_pattern)
        else:
            sync_bits = self.binary_to_bits(sync_pattern)

        if sync_bits is None:
            print("Error: Invalid sync pattern")
            return None

        sync_length = len(sync_bits)
        counter_length = 8 if include_counter else 0

        all_data = []

        for frame_num in range(num_frames):
            # Determine frame width for this frame
            if frame_length_mode == "static":
                frame_width = static_frame_width
            else:  # variable
                frame_width = np.random.randint(min_frame_width, max_frame_width + 1)

            payload_length = frame_width - sync_length - counter_length

            if payload_length <= 0:
                print(
                    f"Error: Frame {frame_num} width ({frame_width}) too small for sync ({sync_length}) + counter ({counter_length})")
                return None

            # Add sync pattern
            all_data.extend(sync_bits)

            # Add counter if enabled
            if include_counter:
                counter_value = frame_num % 256  # 8-bit counter wraps at 256
                counter_bits = np.array([int(b) for b in format(counter_value, '08b')], dtype=np.uint8)
                all_data.extend(counter_bits)

            # Generate payload data
            if data_type == "random":
                payload = self.generate_random_data(payload_length)
            elif data_type == "pattern":
                payload = self.generate_pattern_data(payload_length, data_pattern)
            elif data_type == "zeros":
                payload = np.zeros(payload_length, dtype=np.uint8)
            elif data_type == "ones":
                payload = np.ones(payload_length, dtype=np.uint8)
            elif data_type == "alternating":
                payload = np.array([i % 2 for i in range(payload_length)], dtype=np.uint8)
            else:
                payload = self.generate_random_data(payload_length)

            all_data.extend(payload)

            if frame_num < 5 or frame_num % max(1, num_frames // 10) == 0:
                frame_start = len(all_data) - len(payload) - counter_length - sync_length
                counter_info = f", counter={frame_num % 256}" if include_counter else ""
                print(f"  Frame {frame_num + 1}: sync at bit {frame_start}, width={frame_width}{counter_info}")

        all_data = np.array(all_data, dtype=np.uint8)

        # Apply LRS if requested
        if lrs_mode and lrs_taps:
            print("\nApplying LRS scrambling...")
            if lrs_mode == "additive":
                all_data = self.apply_additive_lrs(all_data, lrs_taps, lrs_initial_state)
            elif lrs_mode == "feedthrough":
                all_data = self.apply_feedthrough_lrs(all_data, lrs_taps)

        return all_data

    def show_preview(self, data: np.ndarray, sync_pattern: str, include_counter: bool):
        """Show preview of generated data"""
        print(f"\nPreview of generated data:")
        print(f"Total bits: {len(data)}")

        # Convert sync pattern for comparison
        if sync_pattern.startswith('0x') or sync_pattern.startswith('0X'):
            sync_bits = self.hex_to_bits(sync_pattern)
        else:
            sync_bits = self.binary_to_bits(sync_pattern)

        sync_len = len(sync_bits)

        # Show first frame structure
        print(f"\nFirst frame structure:")

        # Show sync part
        sync_part = data[:sync_len]
        sync_str = ''.join(str(b) for b in sync_part)
        print(f"  Sync ({sync_len} bits): {sync_str}")

        # Show counter if present
        offset = sync_len
        if include_counter:
            counter_part = data[sync_len:sync_len + 8]
            counter_str = ''.join(str(b) for b in counter_part)
            counter_val = int(counter_str, 2)
            print(f"  Counter (8 bits): {counter_str} (decimal: {counter_val})")
            offset += 8

        # Show payload part
        payload_preview_len = min(32, len(data) - offset)
        if payload_preview_len > 0:
            payload_part = data[offset:offset + payload_preview_len]
            payload_str = ''.join(str(b) for b in payload_part)
            if len(data) - offset > 32:
                payload_str += "..."
            print(f"  Payload (first {payload_preview_len} bits): {payload_str}")

    def interactive_session(self):
        """Main interactive session for frame generation"""
        print("=" * 60)
        print("Interactive Frame Data Generator")
        print("=" * 60)
        print("This tool creates binary files with structured frames and sync patterns.\n")

        # Get filename
        filename = self.ask_input("What should the output file be called?", "framed_data.bin")
        if not filename.endswith('.bin'):
            filename += '.bin'

        # Get sync pattern
        print("\n--- Sync Pattern ---")
        sync_pattern = self.ask_input("Sync pattern (hex like 0x47 or binary like 00110111)", "0x47")

        # Validate sync pattern
        if sync_pattern.startswith('0x') or sync_pattern.startswith('0X'):
            test_sync = self.hex_to_bits(sync_pattern)
        else:
            test_sync = self.binary_to_bits(sync_pattern)

        if test_sync is None:
            print("Invalid sync pattern, using default 0x47")
            sync_pattern = "0x47"
            test_sync = self.hex_to_bits(sync_pattern)

        sync_len = len(test_sync)
        print(f"Sync pattern length: {sync_len} bits")

        # Ask about 8-bit counter
        print("\n--- Frame Counter ---")
        include_counter = self.ask_yes_no("Include 8-bit counter after sync?")
        counter_len = 8 if include_counter else 0

        # Get frame length mode
        print("\n--- Frame Length ---")
        print("Frame length mode:")
        print("1. Static (all frames same length)")
        print("2. Variable (random length per frame)")
        length_mode = self.ask_integer("Choose mode (1 or 2)", 1, 2, 1)

        frame_length_mode = "static" if length_mode == 1 else "variable"
        static_frame_width = None
        min_frame_width = None
        max_frame_width = None

        if frame_length_mode == "static":
            static_frame_width = self.ask_integer("Frame width in bits?", sync_len + counter_len + 1, 100000, 188)
            payload_len = static_frame_width - sync_len - counter_len
            print(
                f"Frame structure: {sync_len}-bit sync + {counter_len}-bit counter + {payload_len}-bit payload = {static_frame_width} bits")
        else:
            min_frame_width = self.ask_integer("Minimum frame width in bits?", sync_len + counter_len + 1, 100000, 100)
            max_frame_width = self.ask_integer("Maximum frame width in bits?", min_frame_width, 100000, 200)
            print(f"Frames will vary from {min_frame_width} to {max_frame_width} bits")

        # Get number of frames
        num_frames = self.ask_integer("How many frames to generate?", 1, 100000, 100)

        # Get payload data type
        print("\n--- Payload Data ---")
        print("What type of payload data?")
        print("1. Random data")
        print("2. Pattern (repeating)")
        print("3. All zeros")
        print("4. All ones")
        print("5. Alternating (010101...)")

        data_choice = self.ask_integer("Choose payload type (1-5)", 1, 5, 1)

        data_type_map = {
            1: "random",
            2: "pattern",
            3: "zeros",
            4: "ones",
            5: "alternating"
        }

        data_type = data_type_map[data_choice]
        data_pattern = None

        if data_type == "pattern":
            data_pattern = self.ask_input("Enter pattern for payload (hex or binary)", "0xAA")

        # Ask about LRS scrambling
        print("\n--- LRS Scrambling (Optional) ---")
        apply_lrs = self.ask_yes_no("Apply LRS scrambling to the data?")

        lrs_mode = None
        lrs_taps = None
        lrs_initial_state = None

        if apply_lrs:
            print("\nLRS Mode:")
            print("1. Additive (autonomous sequence XORed with data)")
            print("2. Feedthrough (data processed through shift register)")
            lrs_mode_choice = self.ask_integer("Choose LRS mode (1 or 2)", 1, 2, 1)

            lrs_mode = "additive" if lrs_mode_choice == 1 else "feedthrough"

            # Get taps
            print("\nCommon LRS polynomials:")
            print("  7-bit:  0,1,7  or  0,3,7")
            print("  8-bit:  0,2,3,4,8")
            print("  9-bit:  0,4,9")
            print("  15-bit: 0,1,15")

            taps_str = self.ask_input("Enter tap positions (comma or space separated)", "0,1,7")
            lrs_taps = self.parse_taps(taps_str)

            if lrs_taps is None or len(lrs_taps) < 2:
                print("Invalid taps, disabling LRS")
                apply_lrs = False
            else:
                max_tap = max(lrs_taps)
                reg_size = max_tap + 1
                period = (2 ** reg_size) - 1

                print(f"Register size: {reg_size} bits")
                print(f"Period: {period} bits")

                if lrs_mode == "additive":
                    use_default_state = self.ask_yes_no(f"Use default initial state (all 1's)?")

                    if not use_default_state:
                        lrs_initial_state = self.ask_input(
                            f"Enter initial state ({reg_size} bits, hex or binary)",
                            "1" * reg_size
                        )

        # Generate the data
        print(f"\nGenerating {num_frames} frames...")

        generated_data = self.create_framed_data(
            sync_pattern, num_frames, frame_length_mode, static_frame_width,
            min_frame_width, max_frame_width, include_counter, data_type, data_pattern,
            lrs_mode, lrs_taps, lrs_initial_state
        )

        if generated_data is None:
            print("Failed to generate data")
            return

        # Show preview
        self.show_preview(generated_data, sync_pattern, include_counter)

        # Calculate statistics
        total_bits = len(generated_data)
        total_bytes = (total_bits + 7) // 8
        ones = np.sum(generated_data)

        print(f"\nStatistics:")
        print(f"  Total bits: {total_bits}")
        print(f"  Total bytes: {total_bytes}")
        print(f"  Frames: {num_frames}")
        print(f"  Ones ratio: {ones / total_bits * 100:.1f}%")

        if lrs_mode:
            print(f"\nLRS Configuration:")
            print(f"  Mode: {lrs_mode.capitalize()}")
            print(f"  Taps: {lrs_taps}")
            if lrs_mode == "additive" and lrs_initial_state:
                print(f"  Initial state: {lrs_initial_state}")

        # Save the file
        if self.save_file(generated_data, filename):
            print(f"\nSuccessfully created {filename}!")
            print(f"You can now process this file with your bit processing tools.")

            if lrs_mode:
                print(f"\nTo descramble, use LRS tool with:")
                print(f"  Mode: {lrs_mode}")
                print(f"  Taps: {','.join(str(t) for t in lrs_taps)}")
        else:
            print("\nFailed to save file")


def main():
    generator = FrameDataGenerator()
    generator.interactive_session()


if __name__ == '__main__':
    main()

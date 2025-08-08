#!/usr/bin/env python3
"""
PS1 Mask Analysis Utilities

Tools for analyzing and understanding PS1 bitmask combinations using REAL PS1 definitions.
"""

import numpy as np
from ps1_mask_definitions import PS1_MASK_BITS, PS1_SPECIAL_VALUES, decode_ps1_mask, encode_ps1_mask

def analyze_combined_mask(mask_array: np.ndarray, sample_pixels: int = 1000):
    """
    Analyze a combined PS1 mask to understand flag usage.
    
    Args:
        mask_array: Combined PS1 mask array (uint32)
        sample_pixels: Number of pixels to sample for detailed analysis
        
    Returns:
        Dictionary with mask statistics
    """
    print("🔍 PS1 Mask Analysis (Real PS1 Definitions)")
    print("=" * 60)
    
    # Basic statistics
    total_pixels = mask_array.size
    masked_pixels = np.count_nonzero(mask_array)
    clean_pixels = total_pixels - masked_pixels
    
    print(f"📊 Basic Statistics:")
    print(f"   Total pixels: {total_pixels:,}")
    print(f"   Clean pixels: {clean_pixels:,} ({clean_pixels/total_pixels*100:.1f}%)")
    print(f"   Masked pixels: {masked_pixels:,} ({masked_pixels/total_pixels*100:.1f}%)")
    
    # Flag frequency analysis
    print(f"\n🏷️  Flag Frequency Analysis:")
    flag_counts = {}
    
    for flag_name, bit_value in PS1_MASK_BITS.items():
        # Count pixels with this specific flag set
        count = np.count_nonzero(mask_array & bit_value)
        flag_counts[flag_name] = count
        percentage = count / total_pixels * 100
        print(f"   {flag_name:12}: {count:8,} pixels ({percentage:5.2f}%)")
    
    # Special values analysis
    print(f"\n⭐ Special Values Analysis:")
    for special_name, special_value in PS1_SPECIAL_VALUES.items():
        count = np.count_nonzero(mask_array == special_value)
        percentage = count / total_pixels * 100
        print(f"   {special_name:12}: {count:8,} pixels ({percentage:5.2f}%)")
    
    # Most common mask values
    print(f"\n📈 Most Common Mask Values:")
    unique_values, counts = np.unique(mask_array, return_counts=True)
    
    # Sort by frequency and show top 10
    sorted_indices = np.argsort(counts)[::-1]
    for i, idx in enumerate(sorted_indices[:10]):
        value = unique_values[idx]
        count = counts[idx]
        percentage = count / total_pixels * 100
        
        if value == 0:
            flags_str = "CLEAN"
        else:
            flags = decode_ps1_mask(value)
            flags_str = " + ".join(flags) if flags else f"UNKNOWN({value})"
        
        print(f"   {i+1:2d}. Value {value:5d}: {count:8,} pixels ({percentage:5.2f}%) - {flags_str}")
    
    # Sample detailed analysis
    if masked_pixels > 0:
        print(f"\n🔬 Sample Pixel Analysis (random {min(sample_pixels, masked_pixels)} masked pixels):")
        masked_indices = np.where(mask_array != 0)
        sample_size = min(sample_pixels, len(masked_indices[0]))
        sample_indices = np.random.choice(len(masked_indices[0]), sample_size, replace=False)
        
        for i in range(min(10, sample_size)):  # Show first 10 samples
            idx = sample_indices[i]
            y, x = masked_indices[0][idx], masked_indices[1][idx]
            value = mask_array[y, x]
            flags = decode_ps1_mask(value)
            print(f"   Pixel ({y:4d},{x:4d}): value={value:5d} -> {' + '.join(flags) if flags else 'UNKNOWN'}")
    
    return {
        'total_pixels': total_pixels,
        'clean_pixels': clean_pixels,
        'masked_pixels': masked_pixels,
        'flag_counts': flag_counts,
        'unique_values': unique_values,
        'value_counts': counts
    }

def create_flag_specific_mask(combined_mask: np.ndarray, flags: list) -> np.ndarray:
    """
    Create a binary mask for specific PS1 flags.
    
    Args:
        combined_mask: Combined PS1 mask array
        flags: List of flag names to include
        
    Returns:
        Binary mask where True indicates pixels with any of the specified flags
    """
    flag_mask = np.zeros_like(combined_mask, dtype=bool)
    
    for flag in flags:
        if flag in PS1_MASK_BITS:
            bit_value = PS1_MASK_BITS[flag]
            flag_mask |= (combined_mask & bit_value) != 0
        elif flag in PS1_SPECIAL_VALUES:
            special_value = PS1_SPECIAL_VALUES[flag]
            flag_mask |= (combined_mask == special_value)
    
    return flag_mask

def mask_quality_report(combined_mask: np.ndarray):
    """Generate a comprehensive quality report for the combined mask."""
    
    print("📋 PS1 Mask Quality Report (Real PS1 Definitions)")
    print("=" * 70)
    
    # Analyze overall mask
    stats = analyze_combined_mask(combined_mask)
    
    # Critical issues analysis
    print(f"\n⚠️  Critical Issues:")
    critical_flags = ['DETECTOR', 'SAT', 'BLANK', 'CR']
    critical_mask = create_flag_specific_mask(combined_mask, critical_flags)
    critical_count = np.count_nonzero(critical_mask)
    print(f"   Pixels with critical issues: {critical_count:,} ({critical_count/combined_mask.size*100:.2f}%)")
    
    # Calibration quality analysis  
    print(f"\n� Calibration Quality:")
    cal_flags = ['FLAT', 'DARK', 'CTE']
    cal_mask = create_flag_specific_mask(combined_mask, cal_flags)
    cal_count = np.count_nonzero(cal_mask)
    print(f"   Calibration issues: {cal_count:,} ({cal_count/combined_mask.size*100:.2f}%)")
    
    # Science quality analysis
    print(f"\n🔬 Science Quality:")
    suspect_mask = create_flag_specific_mask(combined_mask, ['SUSPECT'])
    suspect_count = np.count_nonzero(suspect_mask)
    print(f"   Suspect pixels: {suspect_count:,} ({suspect_count/combined_mask.size*100:.2f}%)")
    
    low_mask = create_flag_specific_mask(combined_mask, ['LOW'])
    low_count = np.count_nonzero(low_mask)
    print(f"   Low signal pixels: {low_count:,} ({low_count/combined_mask.size*100:.2f}%)")
    
    # Artifacts analysis
    print(f"\n👻 Artifact Detection:")
    ghost_mask = create_flag_specific_mask(combined_mask, ['GHOST'])
    ghost_count = np.count_nonzero(ghost_mask)
    print(f"   Ghost/reflection: {ghost_count:,} ({ghost_count/combined_mask.size*100:.2f}%)")
    
    spike_mask = create_flag_specific_mask(combined_mask, ['SPIKE'])
    spike_count = np.count_nonzero(spike_mask)
    print(f"   Spikes/glitches: {spike_count:,} ({spike_count/combined_mask.size*100:.2f}%)")
    
    streak_mask = create_flag_specific_mask(combined_mask, ['STREAK'])
    streak_count = np.count_nonzero(streak_mask)
    print(f"   Streaks: {streak_count:,} ({streak_count/combined_mask.size*100:.2f}%)")
    
    # Star-related flags
    print(f"\n⭐ Star-related Flags:")
    starcore_mask = create_flag_specific_mask(combined_mask, ['STARCORE'])
    starcore_count = np.count_nonzero(starcore_mask)
    print(f"   Star cores: {starcore_count:,} ({starcore_count/combined_mask.size*100:.2f}%)")
    
    # Convolution quality
    print(f"\n🔄 Convolution Quality:")
    conv_bad_mask = create_flag_specific_mask(combined_mask, ['CONV.BAD'])
    conv_bad_count = np.count_nonzero(conv_bad_mask)
    print(f"   Bad convolution: {conv_bad_count:,} ({conv_bad_count/combined_mask.size*100:.2f}%)")
    
    conv_poor_mask = create_flag_specific_mask(combined_mask, ['CONV.POOR'])
    conv_poor_count = np.count_nonzero(conv_poor_mask)
    print(f"   Poor convolution: {conv_poor_count:,} ({conv_poor_count/combined_mask.size*100:.2f}%)")

if __name__ == "__main__":
    # Example usage
    print("🧪 PS1 Mask Analysis Example (Real PS1 Definitions)")
    print("Load a mask file and run:")
    print("   from astropy.io import fits")
    print("   with fits.open('mask_file.fits') as hdul:")
    print("       mask_data = hdul[0].data")
    print("       analyze_combined_mask(mask_data)")

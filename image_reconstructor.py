import numpy as np
import os
import json
from scipy.spatial import cKDTree
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import cv2
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot

@dataclass
class ImageSegment:
    """Class to store information about each image segment"""
    index: int
    center_x: float
    center_y: float
    image: np.ndarray

@dataclass
class InferredParameters:
    """Enhanced class to store inferred parameters from segment positions"""
    position_uncertainty: float
    grid_spacing_x: float      # Separate spacing for x and y directions
    grid_spacing_y: float
    grid_regularity: float     # Measure of how regular the grid structure is
    local_distortions: List[Tuple[int, float]]  # Areas with significant deviation
    confidence_score: float

class AdaptiveImageReconstructor:
    def __init__(self, output_shape: Tuple[int, int]):
        """Initialize the reconstructor"""
        self.output_shape = output_shape
        self.segment_size = 600
        self.inferred_params = None
        self.min_overlap = 0.02  # Minimum expected overlap
        self.max_overlap = 0.25  # Maximum expected overlap

    def _find_segment_neighbors(self, segments: List[ImageSegment]) -> Dict[int, List[Tuple[int, float, float]]]:
        """Enhanced neighbor finding with directional analysis"""
        centers = np.array([(seg.center_x, seg.center_y) for seg in segments])
        tree = cKDTree(centers)
        
        # Increased search radius to account for variable overlap
        max_spacing = self.segment_size * (1 - self.min_overlap)
        neighbors = tree.query_ball_tree(tree, r=max_spacing * 1.2)
        
        neighbor_dict = defaultdict(list)
        for i, neighbor_indices in enumerate(neighbors):
            for j in neighbor_indices:
                if i != j:
                    # Calculate vector between centers
                    vector = centers[j] - centers[i]
                    distance = np.linalg.norm(vector)
                    angle = np.arctan2(vector[1], vector[0])
                    
                    neighbor_dict[i].append((j, distance, angle))
        
        return neighbor_dict

    def _analyze_local_patterns(self, segments: List[ImageSegment], 
                              neighbor_dict: Dict[int, List[Tuple[int, float, float]]]) -> dict:
        """Analyze local patterns in segment arrangement"""
        local_patterns = {
            'spacings_x': [],
            'spacings_y': [],
            'angle_deviations': [],
            'local_distortions': []
        }
        
        for idx, neighbors in neighbor_dict.items():
            for neighbor_idx, distance, angle in neighbors:
                # Normalize angle to principal directions
                norm_angle = abs(angle % (np.pi/2))
                if norm_angle > np.pi/4:
                    norm_angle = np.pi/2 - norm_angle
                
                # Record spacing based on direction
                if norm_angle < np.pi/8:  # Horizontal
                    local_patterns['spacings_x'].append(distance)
                else:  # Vertical
                    local_patterns['spacings_y'].append(distance)
                
                local_patterns['angle_deviations'].append(norm_angle)
                
                # Detect local distortions
                if abs(distance - np.mean(local_patterns['spacings_x'] + local_patterns['spacings_y'])) > \
                   2 * np.std(local_patterns['spacings_x'] + local_patterns['spacings_y']):
                    local_patterns['local_distortions'].append((idx, distance))
        
        return local_patterns

    def _calculate_confidence_score(self, patterns: dict) -> float:
        """Calculate confidence score based on pattern analysis"""
        scores = []
        
        # Check spacing consistency
        if len(patterns['spacings_x']) > 0:
            spacing_x_cv = np.std(patterns['spacings_x']) / np.mean(patterns['spacings_x'])
            scores.append(np.exp(-spacing_x_cv))
        
        if len(patterns['spacings_y']) > 0:
            spacing_y_cv = np.std(patterns['spacings_y']) / np.mean(patterns['spacings_y'])
            scores.append(np.exp(-spacing_y_cv))
        
        # Check grid regularity
        if patterns['angle_deviations']:
            angle_score = 1 - np.mean(patterns['angle_deviations']) / (np.pi/4)
            scores.append(angle_score)
        
        return np.mean(scores) if scores else 0.0

    def infer_parameters(self, segments: List[ImageSegment]) -> InferredParameters:
        """Infer reconstruction parameters with enhanced pattern recognition"""
        neighbor_dict = self._find_segment_neighbors(segments)
        patterns = self._analyze_local_patterns(segments, neighbor_dict)
        
        # Calculate grid spacings
        grid_spacing_x = np.median(patterns['spacings_x']) if patterns['spacings_x'] else self.segment_size
        grid_spacing_y = np.median(patterns['spacings_y']) if patterns['spacings_y'] else self.segment_size
        
        # Calculate position uncertainty
        position_diffs = []
        for idx, neighbors in neighbor_dict.items():
            for _, distance, _ in neighbors:
                expected_distance = min(grid_spacing_x, grid_spacing_y)
                position_diffs.append(abs(distance - expected_distance))
        
        position_uncertainty = np.std(position_diffs) if position_diffs else 10.0
        
        # Calculate grid regularity
        grid_regularity = 1.0 - np.std(patterns['angle_deviations']) / (np.pi/4) \
            if patterns['angle_deviations'] else 0.0
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(patterns)
        
        self.inferred_params = InferredParameters(
            position_uncertainty=position_uncertainty,
            grid_spacing_x=grid_spacing_x,
            grid_spacing_y=grid_spacing_y,
            grid_regularity=grid_regularity,
            local_distortions=patterns['local_distortions'],
            confidence_score=confidence_score
        )
        
        return self.inferred_params

    def _calculate_adaptive_weights(self, segment: ImageSegment, x: int, y: int,
                                 local_distortion: bool = False) -> np.ndarray:
        """Calculate adaptive weights based on local conditions"""
        y_grid, x_grid = np.ogrid[-300:300, -300:300]
        
        # Adjust sigma based on position uncertainty and local distortions
        base_sigma = max(200, self.inferred_params.position_uncertainty * 10)
        if local_distortion:
            base_sigma *= 1.5
        
        # Use uniform sigma for weighting
        sigma_x = base_sigma
        sigma_y = base_sigma
        
        weights = np.exp(-(x_grid**2/(2*sigma_x**2) + y_grid**2/(2*sigma_y**2)))
        
        return weights

    def reconstruct(self, segments: List[ImageSegment]) -> np.ndarray:
        """Reconstruct image with adaptive blending"""
        if self.inferred_params is None:
            self.infer_parameters(segments)
        
        output = np.zeros((*self.output_shape, 3), dtype=np.float32)
        weights = np.zeros(self.output_shape, dtype=np.float32)
        
        # Create set of segments with local distortions
        distorted_segments = {idx for idx, _ in self.inferred_params.local_distortions}
        
        # Process each segment with adaptive weighting
        for segment in segments:
            x = int(segment.center_x - self.segment_size/2)
            y = int(segment.center_y - self.segment_size/2)
            
            # Calculate valid regions
            y_start = max(0, y)
            y_end = min(self.output_shape[0], y + self.segment_size)
            x_start = max(0, x)
            x_end = min(self.output_shape[1], x + self.segment_size)
            
            seg_y_start = max(0, -y)
            seg_y_end = self.segment_size - max(0, y + self.segment_size - self.output_shape[0])
            seg_x_start = max(0, -x)
            seg_x_end = self.segment_size - max(0, x + self.segment_size - self.output_shape[1])
            
            if (seg_y_end - seg_y_start <= 0) or (seg_x_end - seg_x_start <= 0):
                continue
            
            try:
                # Get appropriate regions
                segment_region = segment.image[seg_y_start:seg_y_end, seg_x_start:seg_x_end]
                weight_region = self._calculate_adaptive_weights(
                    segment, x, y,
                    local_distortion=(segment.index in distorted_segments)
                )[seg_y_start:seg_y_end, seg_x_start:seg_x_end]
                
                if segment_region.shape[:2] != weight_region.shape:
                    continue
                
                # Apply confidence-based weighting
                weight_region *= self.inferred_params.confidence_score
                
                # Add weighted contribution
                output[y_start:y_end, x_start:x_end] += segment_region * weight_region[..., np.newaxis]
                weights[y_start:y_end, x_start:x_end] += weight_region
                
            except ValueError as e:
                print(f"Skipping segment {segment.index} due to error: {e}")
                continue
        
        # Normalize and handle gaps
        mask = weights > 0
        output[mask] /= weights[mask][..., np.newaxis]
        
        # Fill gaps using nearest neighbor interpolation
        if not np.all(mask):
            for c in range(3):
                channel = output[..., c]
                valid_mask = mask
                coords = np.array(np.nonzero(valid_mask)).T
                values = channel[valid_mask]
                
                missing_coords = np.array(np.nonzero(~valid_mask)).T
                
                if len(missing_coords) > 0 and len(coords) > 0:
                    tree = cKDTree(coords)
                    _, nearest_idx = tree.query(missing_coords)
                    channel[~valid_mask] = values[nearest_idx]
        
        return np.clip(output, 0, 255).astype(np.uint8)

def load_segments(input_dir='segments'):
    # Load metadata
    with open(os.path.join(input_dir, 'segments_metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    # Reconstruct segments
    segments = []
    for meta in metadata:
        segment_path = os.path.join(input_dir, f'segment_{meta["index"]}.npy')
        image = np.load(segment_path)
        
        segments.append(ImageSegment(
            index=meta['index'],
            center_x=meta['center_x'],
            center_y=meta['center_y'],
            image=image
        ))
    
    return segments

def process_image(segments_dir: str, 
                  output_path: Optional[str] = None, 
                  visualize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Process image reconstruction with robust error handling"""
    try:
        # Load segments with error handling
        try:
            segments = load_segments(segments_dir)
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            print(f"Error loading segments: {e}")
            raise
        
        # Read original image from metadata if possible
        with open(os.path.join(segments_dir, 'segments_metadata.json'), 'r') as f:
            metadata = json.load(f)
            max_height = max(int(meta['center_y'] + 300) for meta in metadata)
            max_width = max(int(meta['center_x'] + 300) for meta in metadata)
        
        # Initialize reconstructor
        reconstructor = AdaptiveImageReconstructor(output_shape=(max_height, max_width))
        
        # Infer parameters
        params = reconstructor.infer_parameters(segments)
        
        # Print inferred parameters
        print("\nInferred Parameters:")
        print(f"Position uncertainty: {params.position_uncertainty:.2f} pixels")
        print(f"Grid spacing X: {params.grid_spacing_x:.2f} pixels")
        print(f"Grid spacing Y: {params.grid_spacing_y:.2f} pixels")
        print(f"Grid regularity: {params.grid_regularity:.2f}")
        print(f"Confidence score: {params.confidence_score:.2f}")
        print(f"Number of local distortions: {len(params.local_distortions)}\n")
        
        # Reconstruct image
        reconstructed = reconstructor.reconstruct(segments)
        
        # Save reconstructed image if output path is provided
        if output_path:
            output_dir = Path(output_path).parent
            output_dir.mkdir(exist_ok=True)
            output_img = cv2.cvtColor(reconstructed, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), output_img)
            
        # Return both original (approximated) and reconstructed image
        return np.zeros_like(reconstructed), reconstructed
    
    except Exception as e:
        print(f"Reconstruction failed: {e}")
        raise

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Process satellite image simulation with parameter inference')
    parser.add_argument('--segments_dir', type=str, default='segments', help='Path to input image')
    parser.add_argument('--output_path', type=str, default='visualizations/reconstructed.png', help='Path to save output image')
    parser.add_argument('--no-viz', action='store_false', dest='visualize',
                       help='Disable visualization')
    
    args = parser.parse_args()
    
    # Process the image
    try:
        original, reconstructed = process_image(
            args.segments_dir,
            args.output_path,
            visualize=args.visualize
        )
        print("Processing completed successfully!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
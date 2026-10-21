/**
 * Type definitions for NeuroScope 3D NIfTI Upload & Validation Module.
 */

export type Modality = 'T1' | 'T1Gd' | 'T2' | 'FLAIR';

export type SlotStatus = 'idle' | 'uploading' | 'validating' | 'valid' | 'invalid';

export interface SlotState {
  modality: Modality;
  file: File | null;
  filename: string;
  sizeMB: number;
  status: SlotStatus;
  uploadProgress: number; // 0-100
  errorMessage: string | null;
  shape?: number[];
  zooms?: number[];
  orientation?: string[];
}

export interface StudyMetadata {
  shape: number[];
  voxel_spacing_mm: number[];
  orientation: string[];
  affine_consistent: boolean;
  all_finite: boolean;
  all_non_constant: boolean;
  total_size_bytes?: number;
  total_size_mb?: number;
  validation_duration_seconds?: number;
}

export interface ValidationSuccessResponse {
  status: 'success';
  valid: true;
  message: string;
  study_id: string;
  modalities: Record<Modality, 'valid'>;
  metadata: StudyMetadata;
  ready_for_segmentation: true;
}

export interface ValidationErrorResponse {
  status: 'error';
  valid: false;
  error_code: string;
  message: string;
  details: Record<string, any>;
  ready_for_segmentation: false;
}

export interface TumorSubRegionMetrics {
  label: string;
  volume_cm3: number;
  voxel_count: number;
}

export interface SegmentationResponse {
  status: 'success';
  study_id: string;
  message: string;
  model: string;
  input_channel_order: string[];
  sub_regions: {
    whole_tumor: TumorSubRegionMetrics;
    tumor_core: TumorSubRegionMetrics;
    enhancing_tumor: TumorSubRegionMetrics;
  };
  whole_tumor_volume_cm3: number;
  tumor_core_volume_cm3: number;
  enhancing_tumor_volume_cm3: number;
  prediction_url: string;
  viewer_url: string;
  report_url: string;
  warnings: string[];
  disclaimer: string;
}

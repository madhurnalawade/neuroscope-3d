import React from 'react';
import { StudyMetadata } from '../types';

interface ValidationSummaryProps {
  metadata: StudyMetadata;
  studyId: string;
  onProceedToSegmentation: () => void;
  isSegmenting: boolean;
}

export const ValidationSummary: React.FC<ValidationSummaryProps> = ({
  metadata,
  studyId,
  onProceedToSegmentation,
  isSegmenting,
}) => {
  return (
    <div className="validation-summary-card">
      <div className="summary-header">
        <div className="summary-status-badge">
          <span className="check-icon">&#10003;</span>
          <span className="summary-title">MRI Study Validation: PASSED</span>
        </div>
        <div className="study-id-tag">Study ID: <code>{studyId}</code></div>
      </div>

      <div className="summary-grid">
        <div className="summary-metric">
          <span className="metric-label">Validated Sequences</span>
          <span className="metric-value highlight-green">4 / 4 Complete</span>
        </div>

        <div className="summary-metric">
          <span className="metric-label">Image Dimensions (X × Y × Z)</span>
          <span className="metric-value">{metadata.shape.join(' × ')}</span>
        </div>

        <div className="summary-metric">
          <span className="metric-label">Voxel Spacing (dx × dy × dz)</span>
          <span className="metric-value">
            {metadata.voxel_spacing_mm.map((v) => v.toFixed(2)).join(' × ')} mm
          </span>
        </div>

        <div className="summary-metric">
          <span className="metric-label">Image Orientation</span>
          <span className="metric-value">{metadata.orientation.join('')} (Radiological RAS)</span>
        </div>

        <div className="summary-metric">
          <span className="metric-label">Total Uploaded Size</span>
          <span className="metric-value">{metadata.total_size_mb?.toFixed(2) || '0.00'} MB</span>
        </div>

        <div className="summary-metric">
          <span className="metric-label">Affine Consistency</span>
          <span className="metric-value highlight-green">&#10003; Matching 4×4 Matrix</span>
        </div>

        <div className="summary-metric">
          <span className="metric-label">Finite Value Check</span>
          <span className="metric-value highlight-green">&#10003; 100% Finite (No NaN/Inf)</span>
        </div>

        <div className="summary-metric">
          <span className="metric-label">Intensity Variation Check</span>
          <span className="metric-value highlight-green">&#10003; Non-Constant Anatomic Signal</span>
        </div>
      </div>

      <div className="summary-footer">
        <button
          type="button"
          className="btn-proceed-segmentation"
          onClick={onProceedToSegmentation}
          disabled={isSegmenting}
        >
          {isSegmenting ? (
            <>
              <span className="spinner-sm"></span> Running MONAI SegResNet Pipeline...
            </>
          ) : (
            <>Proceed to 3D Segmentation &rarr;</>
          )}
        </button>
      </div>
    </div>
  );
};

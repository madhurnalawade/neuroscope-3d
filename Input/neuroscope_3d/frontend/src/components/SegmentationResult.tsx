import React from 'react';
import { SegmentationResponse } from '../types';

interface SegmentationResultProps {
  result: SegmentationResponse;
  onReset: () => void;
}

export const SegmentationResult: React.FC<SegmentationResultProps> = ({ result, onReset }) => {
  return (
    <div className="segmentation-result-card">
      <div className="seg-header">
        <h2 className="seg-title">Suspected tumour regions predicted by the model</h2>
        <span className="model-badge">{result.model}</span>
      </div>

      <p className="seg-desc">
        Estimated volumes (voxel count × voxel volume). Not a diagnosis. Input channel order:{' '}
        <code>{result.input_channel_order.join(' → ')}</code>
      </p>

      {result.warnings && result.warnings.length ? (
        <div className="study-error-alert">
          <div className="alert-content">
            <h3 className="alert-title">Input suitability warnings</h3>
            {result.warnings.map((w, i) => (
              <p className="alert-msg" key={i}>{w}</p>
            ))}
          </div>
        </div>
      ) : null}

      <div className="volumetrics-grid">
        <div className="volumetric-box wt-box">
          <span className="vol-title">Whole Tumor (WT)</span>
          <span className="vol-sub">(Edema + Core + Enhancing)</span>
          <span className="vol-value">{result.whole_tumor_volume_cm3.toFixed(2)} cm³</span>
          <span className="voxel-count">{result.sub_regions.whole_tumor.voxel_count.toLocaleString()} voxels</span>
        </div>

        <div className="volumetric-box tc-box">
          <span className="vol-title">Tumor Core (TC)</span>
          <span className="vol-sub">(Necrotic + Enhancing)</span>
          <span className="vol-value">{result.tumor_core_volume_cm3.toFixed(2)} cm³</span>
          <span className="voxel-count">{result.sub_regions.tumor_core.voxel_count.toLocaleString()} voxels</span>
        </div>

        <div className="volumetric-box et-box">
          <span className="vol-title">Enhancing Tumor (ET)</span>
          <span className="vol-sub">(Label 4)</span>
          <span className="vol-value">{result.enhancing_tumor_volume_cm3.toFixed(2)} cm³</span>
          <span className="voxel-count">{result.sub_regions.enhancing_tumor.voxel_count.toLocaleString()} voxels</span>
        </div>
      </div>

      <div className="seg-footer">
        <a className="btn-primary-action" href={result.viewer_url} style={{ marginRight: '0.75rem', textDecoration: 'none', display: 'inline-block', width: 'auto' }}>
          Open 3D &amp; slice viewer
        </a>
        <a className="btn-secondary" href={result.report_url} target="_blank" style={{ marginRight: '0.75rem', textDecoration: 'none' }}>
          Report
        </a>
        <a className="btn-secondary" href={result.prediction_url} style={{ marginRight: '0.75rem', textDecoration: 'none' }}>
          Download label map (NIfTI)
        </a>
        <button type="button" className="btn-secondary" onClick={onReset}>
          Upload Another MRI Study
        </button>
      </div>
    </div>
  );
};

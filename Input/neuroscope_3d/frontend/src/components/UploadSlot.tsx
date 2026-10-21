import React, { useRef, useState } from 'react';
import { Modality, SlotState } from '../types';

interface UploadSlotProps {
  slot: SlotState;
  allModalities: Modality[];
  usedModalities: Record<string, Modality>;
  onFileSelect: (modality: Modality, file: File) => void;
  onFileRemove: (modality: Modality) => void;
  onModalityChange: (oldMod: Modality, newMod: Modality) => void;
}

export const UploadSlot: React.FC<UploadSlotProps> = ({
  slot,
  allModalities,
  usedModalities,
  onFileSelect,
  onFileRemove,
  onModalityChange,
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onFileSelect(slot.modality, e.dataTransfer.files[0]);
    }
  };

  const getStatusClass = () => {
    switch (slot.status) {
      case 'idle':
        return 'status-idle'; // Gray
      case 'uploading':
        return 'status-uploading'; // Blue
      case 'validating':
        return 'status-validating'; // Yellow
      case 'valid':
        return 'status-valid'; // Green
      case 'invalid':
        return 'status-invalid'; // Red
      default:
        return 'status-idle';
    }
  };

  const getStatusBadge = () => {
    switch (slot.status) {
      case 'idle':
        return <span className="badge badge-gray">No file selected</span>;
      case 'uploading':
        return <span className="badge badge-blue">Uploading {slot.uploadProgress}%</span>;
      case 'validating':
        return (
          <span className="badge badge-yellow">
            <span className="spinner-sm"></span> Validating...
          </span>
        );
      case 'valid':
        return <span className="badge badge-green">&#10003; Valid NIfTI</span>;
      case 'invalid':
        return <span className="badge badge-red">&#10007; Invalid</span>;
    }
  };

  return (
    <div
      className={`upload-slot-card ${getStatusClass()} ${isDragOver ? 'drag-over' : ''}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <div className="slot-header">
        <div className="slot-title-area">
          <span className="slot-modality-label">{slot.modality}</span>
          <span className="slot-seq-desc">
            {slot.modality === 'T1' && 'T1-Weighted Structural'}
            {slot.modality === 'T1Gd' && 'T1 Contrast-Enhanced (T1ce)'}
            {slot.modality === 'T2' && 'T2-Weighted Fluid Sensitive'}
            {slot.modality === 'FLAIR' && 'T2-FLAIR Edema Suppressed'}
          </span>
        </div>
        {getStatusBadge()}
      </div>

      <div className="slot-body">
        {slot.file ? (
          <div className="file-info-container">
            <div className="file-details">
              <span className="file-icon">&#128196;</span>
              <div className="file-texts">
                <span className="filename" title={slot.filename}>
                  {slot.filename}
                </span>
                <span className="filesize">{slot.sizeMB.toFixed(2)} MB</span>
              </div>
            </div>

            {slot.status === 'valid' && slot.shape && (
              <div className="slot-metadata-pills">
                <span className="meta-pill">Shape: {slot.shape.join(' × ')}</span>
                {slot.zooms && (
                  <span className="meta-pill">Spacing: {slot.zooms.map((z) => z.toFixed(1)).join(' × ')} mm</span>
                )}
                {slot.orientation && <span className="meta-pill">Orientation: {slot.orientation.join('')}</span>}
              </div>
            )}

            {slot.status === 'invalid' && slot.errorMessage && (
              <div className="slot-error-banner">
                <span className="error-icon">&#9888;</span>
                <p className="error-text">{slot.errorMessage}</p>
              </div>
            )}

            <div className="slot-actions">
              <button
                type="button"
                className="btn-replace"
                onClick={() => inputRef.current?.click()}
              >
                Replace File
              </button>
              <button
                type="button"
                className="btn-remove"
                onClick={() => onFileRemove(slot.modality)}
              >
                Remove
              </button>
            </div>
          </div>
        ) : (
          <div className="dropzone-empty" onClick={() => inputRef.current?.click()}>
            <span className="drop-icon">&#8682;</span>
            <p className="drop-text">Drag & drop .nii or .nii.gz file here</p>
            <p className="drop-subtext">or click to browse from disk</p>
            <div className="accepted-formats">Supported: .nii, .nii.gz (Max 1 GiB)</div>
          </div>
        )}

        <input
          ref={inputRef}
          type="file"
          accept=".nii,.nii.gz,application/gzip"
          className="hidden-file-input"
          onChange={(e) => {
            if (e.target.files && e.target.files.length > 0) {
              onFileSelect(slot.modality, e.target.files[0]);
              e.target.value = '';
            }
          }}
        />
      </div>

      <div className="slot-footer">
        <label className="modality-select-label">
          <span>Assigned Modality:</span>
          <select
            value={slot.modality}
            onChange={(e) => onModalityChange(slot.modality, e.target.value as Modality)}
            className="modality-dropdown"
          >
            {allModalities.map((m) => (
              <option key={m} value={m}>
                {m} {m === 'T1Gd' ? '(T1ce)' : ''}
              </option>
            ))}
          </select>
        </label>
      </div>
    </div>
  );
};

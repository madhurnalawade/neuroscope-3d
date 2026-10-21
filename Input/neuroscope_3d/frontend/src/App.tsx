import React, { useState } from 'react';
import { Modality, SegmentationResponse, SlotState, ValidationSuccessResponse } from './types';
import { detectModalityFromFilename } from './utils/modalityDetector';
import { UploadSlot } from './components/UploadSlot';
import { ValidationSummary } from './components/ValidationSummary';
import { MedicalDisclaimer } from './components/MedicalDisclaimer';
import { SegmentationResult } from './components/SegmentationResult';

const ALL_MODALITIES: Modality[] = ['T1', 'T1Gd', 'T2', 'FLAIR'];

export const App: React.FC = () => {
  const [slots, setSlots] = useState<Record<Modality, SlotState>>({
    T1: { modality: 'T1', file: null, filename: '', sizeMB: 0, status: 'idle', uploadProgress: 0, errorMessage: null },
    T1Gd: { modality: 'T1Gd', file: null, filename: '', sizeMB: 0, status: 'idle', uploadProgress: 0, errorMessage: null },
    T2: { modality: 'T2', file: null, filename: '', sizeMB: 0, status: 'idle', uploadProgress: 0, errorMessage: null },
    FLAIR: { modality: 'FLAIR', file: null, filename: '', sizeMB: 0, status: 'idle', uploadProgress: 0, errorMessage: null },
  });

  const [isValidatingStudy, setIsValidatingStudy] = useState(false);
  const [studyValidationResult, setStudyValidationResult] = useState<ValidationSuccessResponse | null>(null);
  const [studyErrorMessage, setStudyErrorMessage] = useState<string | null>(null);
  const [studyErrorDetails, setStudyErrorDetails] = useState<any | null>(null);

  const [isSegmenting, setIsSegmenting] = useState(false);
  const [segmentationResult, setSegmentationResult] = useState<SegmentationResponse | null>(null);

  // Validate single file live with the backend
  const validateSingleFileWithBackend = async (modality: Modality, file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('modality', modality);

    try {
      const res = await fetch('/api/upload/validate-single', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (res.ok && data.valid) {
        setSlots((prev) => ({
          ...prev,
          [modality]: {
            ...prev[modality],
            status: 'valid',
            errorMessage: null,
            shape: data.shape,
            zooms: data.zooms,
            orientation: data.orientation,
          },
        }));
      } else {
        setSlots((prev) => ({
          ...prev,
          [modality]: {
            ...prev[modality],
            status: 'invalid',
            errorMessage: data.message || 'Invalid NIfTI file format.',
          },
        }));
      }
    } catch (err: any) {
      setSlots((prev) => ({
        ...prev,
        [modality]: {
          ...prev[modality],
          status: 'invalid',
          errorMessage: 'Server communication error during file validation.',
        },
      }));
    }
  };

  const handleFileSelect = (targetModality: Modality, file: File) => {
    // Reset previous study results
    setStudyValidationResult(null);
    setStudyErrorMessage(null);
    setSegmentationResult(null);

    // Automatic modality recognition
    const detected = detectModalityFromFilename(file.name);
    let assignedModality = targetModality;
    if (detected && detected !== targetModality) {
      assignedModality = detected;
    }

    const sizeMB = file.size / (1024 * 1024);

    setSlots((prev) => ({
      ...prev,
      [assignedModality]: {
        modality: assignedModality,
        file: file,
        filename: file.name,
        sizeMB: sizeMB,
        status: 'validating',
        uploadProgress: 100,
        errorMessage: null,
      },
    }));

    validateSingleFileWithBackend(assignedModality, file);
  };

  const handleFileRemove = (modality: Modality) => {
    setStudyValidationResult(null);
    setStudyErrorMessage(null);
    setSegmentationResult(null);
    setSlots((prev) => ({
      ...prev,
      [modality]: {
        modality,
        file: null,
        filename: '',
        sizeMB: 0,
        status: 'idle',
        uploadProgress: 0,
        errorMessage: null,
      },
    }));
  };

  const handleModalityChange = (oldMod: Modality, newMod: Modality) => {
    if (oldMod === newMod) return;
    setStudyValidationResult(null);
    setStudyErrorMessage(null);

    const oldSlot = slots[oldMod];
    const targetSlot = slots[newMod];

    // Swap files between the two slots
    setSlots((prev) => ({
      ...prev,
      [oldMod]: {
        ...targetSlot,
        modality: oldMod,
      },
      [newMod]: {
        ...oldSlot,
        modality: newMod,
      },
    }));
  };

  // Full 4-sequence study validation
  const handleValidateStudy = async () => {
    // Verify all 4 slots have files
    const missing: Modality[] = [];
    ALL_MODALITIES.forEach((m) => {
      if (!slots[m].file) missing.push(m);
    });

    if (missing.length > 0) {
      setStudyErrorMessage(
        `Please upload all 4 required MRI sequences before validating. Missing: ${missing.join(', ')}.`
      );
      return;
    }

    setIsValidatingStudy(true);
    setStudyErrorMessage(null);
    setStudyErrorDetails(null);
    setStudyValidationResult(null);

    const formData = new FormData();
    formData.append('t1', slots.T1.file!);
    formData.append('t1gd', slots.T1Gd.file!);
    formData.append('t2', slots.T2.file!);
    formData.append('flair', slots.FLAIR.file!);

    try {
      const res = await fetch('/api/upload/validate', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();

      if (res.ok && data.valid) {
        setStudyValidationResult(data);
        // Mark all slots as valid
        setSlots((prev) => {
          const updated = { ...prev };
          ALL_MODALITIES.forEach((m) => {
            updated[m] = { ...updated[m], status: 'valid', errorMessage: null };
          });
          return updated;
        });
      } else {
        setStudyErrorMessage(data.message || 'Study validation failed.');
        setStudyErrorDetails(data.details || null);
        // If an affected modality is identified, flag its slot
        const badMod = data.details?.modality as Modality;
        if (badMod && slots[badMod]) {
          setSlots((prev) => ({
            ...prev,
            [badMod]: {
              ...prev[badMod],
              status: 'invalid',
              errorMessage: data.message,
            },
          }));
        }
      }
    } catch (err: any) {
      setStudyErrorMessage('Failed communicating with upload validation server.');
    } finally {
      setIsValidatingStudy(false);
    }
  };

  // Segmentation execution
  const handleProceedToSegmentation = async () => {
    if (!studyValidationResult || !studyValidationResult.study_id) return;
    setIsSegmenting(true);
    try {
      const res = await fetch('/api/segmentation/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ study_id: studyValidationResult.study_id }),
      });
      const data = await res.json();
      if (res.ok && data.status === 'success') {
        setSegmentationResult(data);
      } else {
        setStudyErrorMessage(data.message || 'Segmentation pipeline error.');
      }
    } catch (e: any) {
      setStudyErrorMessage('Network error while running segmentation.');
    } finally {
      setIsSegmenting(false);
    }
  };

  const handleReset = () => {
    setStudyValidationResult(null);
    setStudyErrorMessage(null);
    setSegmentationResult(null);
    setSlots({
      T1: { modality: 'T1', file: null, filename: '', sizeMB: 0, status: 'idle', uploadProgress: 0, errorMessage: null },
      T1Gd: { modality: 'T1Gd', file: null, filename: '', sizeMB: 0, status: 'idle', uploadProgress: 0, errorMessage: null },
      T2: { modality: 'T2', file: null, filename: '', sizeMB: 0, status: 'idle', uploadProgress: 0, errorMessage: null },
      FLAIR: { modality: 'FLAIR', file: null, filename: '', sizeMB: 0, status: 'idle', uploadProgress: 0, errorMessage: null },
    });
  };

  const allFilesSelected = ALL_MODALITIES.every((m) => slots[m].file !== null);

  return (
    <div className="neuroscope-app">
      <header className="app-header">
        <div className="header-brand">
          <span className="brand-icon">&#129504;</span>
          <div>
            <h1 className="brand-title">NeuroScope 3D</h1>
            <span className="brand-subtitle">NIfTI Upload & Multi-Parametric MRI Validation Module</span>
          </div>
        </div>
        <div className="header-badge">BraTS 2021 Co-Registration Standard</div>
      </header>

      <main className="app-main-content">
        <MedicalDisclaimer />

        {/* 4 Upload Slots */}
        <section className="slots-section">
          <div className="section-header">
            <h2 className="section-title">Required Multi-Parametric MRI Sequences</h2>
            <p className="section-description">
              Upload exactly four co-registered brain MRI sequences (.nii or .nii.gz). Automatic modality recognition
              identifies sequences by standard token boundaries. All scans must share matching 3D dimensions and spatial affines.
            </p>
          </div>

          <div className="upload-slots-grid">
            {ALL_MODALITIES.map((modality) => (
              <UploadSlot
                key={modality}
                slot={slots[modality]}
                allModalities={ALL_MODALITIES}
                usedModalities={slots}
                onFileSelect={handleFileSelect}
                onFileRemove={handleFileRemove}
                onModalityChange={handleModalityChange}
              />
            ))}
          </div>

          {/* Validation Action Area */}
          <div className="validation-actions-bar">
            <button
              type="button"
              className="btn-primary-action"
              onClick={handleValidateStudy}
              disabled={!allFilesSelected || isValidatingStudy}
            >
              {isValidatingStudy ? (
                <>
                  <span className="spinner"></span> Validating Study Consistency...
                </>
              ) : (
                <>Validate 4-Sequence Study</>
              )}
            </button>
          </div>

          {/* Study Error Banner */}
          {studyErrorMessage && (
            <div className="study-error-alert">
              <span className="alert-icon">&#9888;</span>
              <div className="alert-content">
                <h3 className="alert-title">Study Validation Rejected</h3>
                <p className="alert-msg">{studyErrorMessage}</p>
                {studyErrorDetails && (
                  <div className="alert-details">
                    <code>{JSON.stringify(studyErrorDetails, null, 2)}</code>
                  </div>
                )}
              </div>
            </div>
          )}
        </section>

        {/* Validation Summary Card */}
        {studyValidationResult && !segmentationResult && (
          <section className="summary-section">
            <ValidationSummary
              metadata={studyValidationResult.metadata}
              studyId={studyValidationResult.study_id}
              onProceedToSegmentation={handleProceedToSegmentation}
              isSegmenting={isSegmenting}
            />
          </section>
        )}

        {/* Segmentation Result */}
        {segmentationResult && (
          <section className="segmentation-section">
            <SegmentationResult result={segmentationResult} onReset={handleReset} />
          </section>
        )}
      </main>

      <footer className="app-footer">
        <p>&copy; 2026 NeuroScope 3D. Research and teaching prototype. Not for clinical diagnosis.</p>
      </footer>
    </div>
  );
};

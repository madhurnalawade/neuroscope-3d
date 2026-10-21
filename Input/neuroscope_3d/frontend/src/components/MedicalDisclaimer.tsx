import React from 'react';

export const MedicalDisclaimer: React.FC = () => {
  return (
    <div className="medical-disclaimer-banner">
      <div className="disclaimer-badge">MEDICAL SAFETY & RESEARCH NOTICE</div>
      <p className="disclaimer-text">
        <strong>Research and teaching prototype only.</strong> Not for diagnosis or clinical decision-making.
        Use only public, de-identified research data (e.g., BraTS 2021). Structural validation does not imply
        clinical suitability or diagnostic accuracy.
      </p>
    </div>
  );
};

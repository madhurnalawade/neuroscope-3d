import { Modality } from '../types';

const T1GD_REGEX = /(?:^|[_\-\.\s])(t1(?:ce|gd|c|post|\+c|\+gd|\-ce|\-gd|\-post|_ce|_gd|_post))(?:$|[_\-\.\s])/i;
const FLAIR_REGEX = /(?:^|[_\-\.\s])(t2[\-_]?flair|flair)(?:$|[_\-\.\s])/i;
const T2_REGEX = /(?:^|[_\-\.\s])(t2(?:w|_w|\-w|weighted|_weighted|\*|star)?)(?:$|[_\-\.\s])/i;
const T1_REGEX = /(?:^|[_\-\.\s])(t1(?:w|_w|\-w|weighted|_weighted)?)(?:$|[_\-\.\s])/i;

export function detectModalityFromFilename(filename: string): Modality | null {
  if (!filename) return null;

  let base = filename;
  const lower = filename.toLowerCase();
  for (const ext of ['.nii.gz', '.nii', '.gz']) {
    if (lower.endsWith(ext)) {
      base = base.slice(0, -ext.length);
      break;
    }
  }

  // Check specific tokens first
  if (T1GD_REGEX.test(base)) return 'T1Gd';
  if (FLAIR_REGEX.test(base)) return 'FLAIR';
  if (T2_REGEX.test(base)) return 'T2';
  if (T1_REGEX.test(base)) return 'T1';

  return null;
}

from pathlib import Path

import nibabel
import numpy as np
import SimpleITK as sitk

root = Path('/mnt/diag/pi-cai')

src_dir = root / 'src'
out_dir = root / 'nnUNet_raw' / 'Dataset001_csPCa'

img_dir = out_dir / 'imagesTr'
label_dir = out_dir / 'labelsTr'
img_dir.mkdir(parents=True, exist_ok=True)
label_dir.mkdir(parents=True, exist_ok=True)

i = 0
for patient in sorted(src_dir.glob('*')):
    if not patient.is_dir():
        continue

    if len(list(patient.glob('*.nii.gz'))) == 0:
        continue

    # if i == 100:
    #     break

    seg_files = list(sorted((patient.glob('*nii.gz'))))
    if len(seg_files) == 1:
        continue

    img = nibabel.load(seg_files[1])
    data = np.asarray(img.dataobj)
    data[data > 0] = 1

    img = nibabel.load(seg_files[0])
    data2 = np.asarray(img.dataobj)
    data[data2 > 0] = 2

    seg_shape = np.array(data.shape)

    scan_file = next(patient.glob('*_t2w.mha'))
    image = sitk.ReadImage(scan_file)

    img_shape = np.array(image.GetSize())

    if not np.all(seg_shape == img_shape):
        continue

    i += 1
    print(i, patient.name)

    img = nibabel.Nifti1Image(data, img.affine)
    out_file = label_dir / f'lesions_{i:03}.nii.gz'
    nibabel.save(img, out_file)

    out_file = img_dir / f'lesions_{i:03}_0000.nii.gz'
    sitk.WriteImage(image, out_file)


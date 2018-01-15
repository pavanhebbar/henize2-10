# HENIZE 2-10
This project deals with the understanding the nature of Hen 2-10, a dwarf
starburst galaxy. We specifically look into the X-ray spectra of the nucleas to
understand the composition and to figure if it is an AGN or not.

In this file, you will be seeing an history of all commits made into this
repository.

* **First Commit** - Uploading README.md and .gitignore files
* **Reprocessed files** - Reprocessed data files using chandra\_repro tool and
  default options. Then a region consisiting of source and a reference
source was extracted, and it was rebinned to 1/8th of its initial size. Then
this image was convoluted with a gaussian of FWHM=0.25 arcsec(sigma = 0.21
pixels.
	* source\_ref\_evt2.fits - Source+reference
	* sourceref\_halfbin.evt2.fits - Above file binned with 0.5 pixel width
	* sourceref\_rebinnned\_evt2.fits - Binned to 0.125 pixel width
	* sourceref\_conv.fits - Convoluted image.

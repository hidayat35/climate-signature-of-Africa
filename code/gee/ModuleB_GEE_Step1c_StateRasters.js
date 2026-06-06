// ============================================================
// MODULE B - STEP 1c: From-Class State Rasters
// ============================================================
// Generates 6 multi-band rasters showing where each key LC
// class existed at the START of each interval.
//
// These are needed to build proper "stable" reference pixels
// for the attribution analysis (e.g., stable FST = was forest,
// stayed forest).
//
// OUTPUT: 6 multi-band rasters (small, fast)
//   state_FST.tif   — 8 bands: forest mask at start of each interval
//   state_SHR.tif   — 8 bands
//   state_GRS.tif   — 8 bands
//   state_BAL.tif   — 8 bands
//   state_CRP.tif   — 8 bands
//   state_NATURAL.tif — 8 bands (FST|SHR|GRS|BAL — for AGEXPANSION)
//
// Resolution: 1000m (matches your existing transition rasters)
// Task count: 6 tasks
// Expected time: 5-15 minutes each = 30-90 minutes total
// ============================================================

var countries = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017');
var africa = countries.filter(ee.Filter.eq('wld_rgn', 'Africa'));
var africaGeom = africa.geometry();
var exportRegion = africaGeom.bounds();

var annual = ee.ImageCollection('projects/sat-io/open-datasets/GLC-FCS30D/annual');
var fiveyear = ee.ImageCollection('projects/sat-io/open-datasets/GLC-FCS30D/five-years-map');

var fromValues = [10,11,12,20,
                  51,52,61,62,71,72,81,82,91,92,
                  120,121,122,
                  130, 140,
                  181,182,183,184,185,186,187,
                  190,
                  150,152,153,200,201,202,
                  210, 220];
var toValues = [1,1,1,1,
                2,2,2,2,2,2,2,2,2,2,
                3,3,3,
                4, 5,
                6,6,6,6,6,6,6,
                7,
                8,8,8,8,8,8,
                9, 10];

function reclassify(image) {
  return image.remap(fromValues, toValues).rename('LC');
}

function getLCForYear(year) {
  if (year >= 2000) {
    var bandIdx = year - 2000 + 1;
    return reclassify(annual.select('b' + bandIdx).mosaic());
  } else {
    var fyIdx = (year === 1985) ? 1 : (year === 1990) ? 2 : 3;
    return reclassify(fiveyear.select('b' + fyIdx).mosaic());
  }
}

var CRP = 1, FST = 2, SHR = 3, GRS = 4, BAL = 8;

// Same interval start years as transitions (band N = start of interval N)
var intervalStartYears = [
  {year: 1985, bandName: 'i1985_1990'},
  {year: 1990, bandName: 'i1990_1995'},
  {year: 1995, bandName: 'i1995_2000'},
  {year: 2000, bandName: 'i2000_2005'},
  {year: 2005, bandName: 'i2005_2010'},
  {year: 2010, bandName: 'i2010_2015'},
  {year: 2015, bandName: 'i2015_2020'},
  {year: 2020, bandName: 'i2020_2022'}
];

// Function: mask for one or more classes at a given year
function classMaskAtYear(year, classCodes) {
  var lc = getLCForYear(year);
  var mask = ee.Image(0);
  classCodes.forEach(function(c) {
    mask = mask.or(lc.eq(c));
  });
  return mask.toByte();
}

// From-class definitions
var fromStates = [
  {name: 'FST',     classes: [FST],                label: 'Forest'},
  {name: 'SHR',     classes: [SHR],                label: 'Shrubland'},
  {name: 'GRS',     classes: [GRS],                label: 'Grassland'},
  {name: 'BAL',     classes: [BAL],                label: 'Bare'},
  {name: 'CRP',     classes: [CRP],                label: 'Cropland'},
  {name: 'NATURAL', classes: [FST, SHR, GRS, BAL], label: 'Natural (FST|SHR|GRS|BAL)'}
];

// ---------------------------------------------------------------
// Export one multi-band raster per from-class
// ---------------------------------------------------------------
print('=== Generating from-class state rasters ===');

fromStates.forEach(function(state) {
  var bands = intervalStartYears.map(function(iv) {
    return classMaskAtYear(iv.year, state.classes).rename(iv.bandName);
  });
  
  var multiband = ee.Image.cat(bands);
  
  Export.image.toDrive({
    image: multiband.clip(africaGeom),
    description: 'state_' + state.name,
    folder: 'PhD_Paper3_Data',
    fileNamePrefix: 'state_' + state.name,
    region: exportRegion,
    scale: 1000,
    maxPixels: 1e13,
    crs: 'EPSG:4326'
  });
  
  print('  state_' + state.name + '.tif (8 bands): ' + state.label);
});

print('\n=== READY ===');
print('Total tasks: 6');
print('Expected time: 5-15 min each, ~30-90 min total');
print('Go to Tasks tab and run each.');

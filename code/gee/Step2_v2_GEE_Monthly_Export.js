// ============================================================
// STEP 2 (v2 MONTHLY): Export monthly CHIRPS precipitation
// and TerraClimate PET for 1985-2022 as multi-band rasters
// ============================================================
//
// OUTPUTS:
//   - precip_monthly.tif  (456 bands: 38 years x 12 months)
//   - pet_monthly.tif     (456 bands: 38 years x 12 months)
//
// Band naming convention: y{YYYY}_m{MM}  (e.g., y1985_m01, y1985_m02, ...)
//
// Export resolution: 5000 m (matches CHIRPS native ~5 km)
// This keeps file sizes manageable (~2-3 GB each).
//
// EXPECTED TIME:
//   - ~15-30 min per export task
//   - 2 tasks total
// ============================================================

// ---- Region setup ----
var countries = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017');
var africa = countries.filter(ee.Filter.eq('wld_rgn', 'Africa'));
var africaGeom = africa.geometry();
var exportRegion = africaGeom.bounds();

print('Africa bounding box set.');

// ---- Time configuration ----
var START_YEAR = 1985;
var END_YEAR = 2022;
var years = ee.List.sequence(START_YEAR, END_YEAR);
var months = ee.List.sequence(1, 12);

// Helper: zero-pad integer to 2 digits (for band naming)
var pad2 = function(n) {
  return ee.Number(n).format('%02d');
};


// ============================================================
// PART 1: CHIRPS MONTHLY PRECIPITATION
// ============================================================
// CHIRPS is at daily resolution natively; we sum to monthly totals.
// Source: UCSB-CHG/CHIRPS/DAILY
// Coverage: 1981-present at 0.05° (~5.5 km)
// ============================================================

print('=== PART 1: CHIRPS monthly precipitation ===');

var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY');

// Build a list of monthly precipitation images with named bands
var monthlyPrecipList = years.map(function(year) {
  return months.map(function(month) {
    year = ee.Number(year);
    month = ee.Number(month);
    
    var startDate = ee.Date.fromYMD(year, month, 1);
    var endDate = startDate.advance(1, 'month');
    
    var monthlySum = chirps.filterDate(startDate, endDate)
                            .select('precipitation')
                            .sum()
                            .rename(ee.String('y').cat(year.format('%d'))
                                     .cat('_m').cat(pad2(month)));
    return monthlySum;
  });
}).flatten();

// Convert to a multi-band image
var precipMultiband = ee.ImageCollection(monthlyPrecipList).toBands();

// GEE's toBands() prefixes band names with index numbers; rename using band index
// Use band renaming to keep clean names
var origBandNames = precipMultiband.bandNames();
var cleanBandNames = origBandNames.map(function(bn) {
  // Extract the original name after the toBands prefix
  return ee.String(bn).split('_').slice(1).join('_');
});
precipMultiband = precipMultiband.rename(cleanBandNames);

print('CHIRPS multi-band image ready. Number of bands:', precipMultiband.bandNames().size());
print('First 5 band names:', precipMultiband.bandNames().slice(0, 5));

Export.image.toDrive({
  image: precipMultiband.clip(africaGeom).toFloat(),
  description: 'precip_monthly_1985_2022',
  folder: 'PhD_Paper3_Data',
  fileNamePrefix: 'precip_monthly_1985_2022',
  region: exportRegion,
  scale: 5000,
  maxPixels: 1e13,
  crs: 'EPSG:4326'
});
print('CHIRPS export task queued.');


// ============================================================
// PART 2: TerraClimate MONTHLY PET
// ============================================================
// TerraClimate PET is already at monthly resolution.
// Source: IDAHO_EPSCOR/TERRACLIMATE
// Coverage: 1958-present at ~4 km
// Scale factor: 0.1 (band 'pet' is in 0.1*mm units)
// ============================================================

print('=== PART 2: TerraClimate monthly PET ===');

var terraclimate = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE');

// Build list of monthly PET images
var monthlyPETList = years.map(function(year) {
  return months.map(function(month) {
    year = ee.Number(year);
    month = ee.Number(month);
    
    var startDate = ee.Date.fromYMD(year, month, 1);
    var endDate = startDate.advance(1, 'month');
    
    // TerraClimate has one image per month — use first() since filterDate is precise
    var monthlyPET = terraclimate.filterDate(startDate, endDate)
                                  .select('pet')
                                  .first()
                                  .multiply(0.1)  // Apply scale factor -> mm
                                  .rename(ee.String('y').cat(year.format('%d'))
                                          .cat('_m').cat(pad2(month)));
    return monthlyPET;
  });
}).flatten();

var petMultiband = ee.ImageCollection(monthlyPETList).toBands();
var origPETNames = petMultiband.bandNames();
var cleanPETNames = origPETNames.map(function(bn) {
  return ee.String(bn).split('_').slice(1).join('_');
});
petMultiband = petMultiband.rename(cleanPETNames);

print('TerraClimate multi-band image ready. Number of bands:', petMultiband.bandNames().size());
print('First 5 band names:', petMultiband.bandNames().slice(0, 5));

Export.image.toDrive({
  image: petMultiband.clip(africaGeom).toFloat(),
  description: 'pet_monthly_1985_2022',
  folder: 'PhD_Paper3_Data',
  fileNamePrefix: 'pet_monthly_1985_2022',
  region: exportRegion,
  scale: 5000,
  maxPixels: 1e13,
  crs: 'EPSG:4326'
});
print('TerraClimate export task queued.');


// ============================================================
// PART 3: Quick visual verification
// ============================================================
print('=== PART 3: Map preview ===');

Map.centerObject(africaGeom, 3);

// Show one month of precip (January 2000) as sanity check
var jan2000Precip = ee.Image(monthlyPrecipList.get(15*12));  // index 180 = 2000, Jan
Map.addLayer(jan2000Precip.clip(africaGeom),
             {min: 0, max: 200, palette: ['white', 'lightblue', 'blue', 'darkblue']},
             'Precip Jan 2000 (mm)', false);

// Show one month of PET (January 2000)
var jan2000PET = ee.Image(monthlyPETList.get(15*12));
Map.addLayer(jan2000PET.clip(africaGeom),
             {min: 0, max: 300, palette: ['white', 'yellow', 'orange', 'red']},
             'PET Jan 2000 (mm)', false);


// ============================================================
// SUMMARY
// ============================================================
print('========================================');
print('MONTHLY CLIMATE EXPORT TASKS READY');
print('========================================');
print('Total tasks: 2');
print('  1. precip_monthly_1985_2022 (456 bands, ~2-3 GB)');
print('  2. pet_monthly_1985_2022 (456 bands, ~2-3 GB)');
print('');
print('Expected runtime: 15-30 min per task');
print('');
print('NEXT STEPS:');
print('  1. Go to Tasks tab and click Run on both tasks');
print('  2. Wait for downloads to Drive');
print('  3. Copy .tif files to:');
print('     D:\\Claude idea\\PhD_Paper3_Data\\monthly_inputs\\');
print('  4. Run Step5_v5_monthly_SPEI.py (to be created next)');

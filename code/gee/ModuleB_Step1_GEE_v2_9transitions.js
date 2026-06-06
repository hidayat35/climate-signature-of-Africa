// ============================================================
// MODULE B - STEP 1 (v2): Generate LC Transition Rasters
// ============================================================
// UPDATED: Now uses the 9-transition framework organized into
// 3 process categories (Degradation / Recovery / Agricultural)
//
// IMPORTANT: If you already started the previous 6-transition
// version's tasks, CANCEL THEM in the Tasks tab before running
// this, to avoid conflicts.
//
// OUTPUTS: 9 cumulative + 9*8 = 72 per-interval = 81 total tasks
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

// LC class codes after reclassification
var CRP = 1, FST = 2, SHR = 3, GRS = 4, TUD = 5;
var WET = 6, IMP = 7, BAL = 8, WTR = 9, PSI = 10;

// ---------------------------------------------------------------
// 9-TRANSITION FRAMEWORK (3 categories)
// ---------------------------------------------------------------
var transitions = [
  // DEGRADATION (climate expected to intensify)
  {name: 'FST_SHR',     category: 'deg', from: [FST], to: [SHR], label: 'Forest degradation'},
  {name: 'SHR_GRS',     category: 'deg', from: [SHR], to: [GRS], label: 'Shrubland degradation'},
  {name: 'FST_CRP',     category: 'deg', from: [FST], to: [CRP], label: 'Deforestation'},
  {name: 'GRS_BAL',     category: 'deg', from: [GRS], to: [BAL], label: 'Desertification'},
  
  // RECOVERY / GREENING (climate may offset)
  {name: 'SHR_FST',     category: 'rec', from: [SHR], to: [FST], label: 'Forest recovery'},
  {name: 'GRS_SHR',     category: 'rec', from: [GRS], to: [SHR], label: 'Woody encroachment'},
  {name: 'BAL_GRS',     category: 'rec', from: [BAL], to: [GRS], label: 'Bare-to-grassland recovery'},
  
  // AGRICULTURAL DYNAMICS (mixed drivers)
  {name: 'AGEXPANSION', category: 'agr', from: [SHR, GRS, BAL], to: [CRP],
   label: 'Agricultural expansion'},
  {name: 'CRP_ABANDONMENT', category: 'agr', from: [CRP], to: [SHR, FST, GRS],
   label: 'Cropland abandonment'}
];

var intervals = [
  {startYear: 1985, endYear: 1990, label: '1985_1990'},
  {startYear: 1990, endYear: 1995, label: '1990_1995'},
  {startYear: 1995, endYear: 2000, label: '1995_2000'},
  {startYear: 2000, endYear: 2005, label: '2000_2005'},
  {startYear: 2005, endYear: 2010, label: '2005_2010'},
  {startYear: 2010, endYear: 2015, label: '2010_2015'},
  {startYear: 2015, endYear: 2020, label: '2015_2020'},
  {startYear: 2020, endYear: 2022, label: '2020_2022'}
];

// ---------------------------------------------------------------
// Transition map function
// ---------------------------------------------------------------
function createTransitionMap(startYear, endYear, fromClasses, toClasses) {
  var lcStart = getLCForYear(startYear);
  var lcEnd = getLCForYear(endYear);
  
  var fromMask = ee.Image(0);
  fromClasses.forEach(function(cls) {
    fromMask = fromMask.or(lcStart.eq(cls));
  });
  
  var toMask = ee.Image(0);
  toClasses.forEach(function(cls) {
    toMask = toMask.or(lcEnd.eq(cls));
  });
  
  return fromMask.and(toMask).rename('transition').toByte();
}


// ---------------------------------------------------------------
// PART 1: Per-interval rasters (9 transitions x 8 intervals = 72)
// ---------------------------------------------------------------
print('=== PART 1: Per-interval transitions (72 tasks) ===');

transitions.forEach(function(t) {
  intervals.forEach(function(i) {
    var transMap = createTransitionMap(i.startYear, i.endYear, t.from, t.to);
    
    Export.image.toDrive({
      image: transMap.clip(africaGeom),
      description: 'transition_' + t.name + '_' + i.label,
      folder: 'PhD_Paper3_Data',
      fileNamePrefix: 'transition_' + t.name + '_' + i.label,
      region: exportRegion,
      scale: 300,
      maxPixels: 1e13,
      crs: 'EPSG:4326'
    });
  });
});

print('  ' + transitions.length + ' transitions x ' + intervals.length + ' intervals = ' +
      (transitions.length * intervals.length));


// ---------------------------------------------------------------
// PART 2: Cumulative rasters (9 tasks)
// ---------------------------------------------------------------
print('\n=== PART 2: Cumulative transitions 1985-2022 (9 tasks) ===');

transitions.forEach(function(t) {
  var cumulative = ee.Image(0);
  intervals.forEach(function(i) {
    cumulative = cumulative.or(
      createTransitionMap(i.startYear, i.endYear, t.from, t.to)
    );
  });
  
  Export.image.toDrive({
    image: cumulative.rename('transition_cumulative').toByte().clip(africaGeom),
    description: 'transition_' + t.name + '_cumulative',
    folder: 'PhD_Paper3_Data',
    fileNamePrefix: 'transition_' + t.name + '_cumulative',
    region: exportRegion,
    scale: 300,
    maxPixels: 1e13,
    crs: 'EPSG:4326'
  });
  
  print('  ' + t.name + ' (' + t.label + ', ' + t.category + ')');
});


// ---------------------------------------------------------------
// PART 3: CATEGORY COMPOSITES (3 tasks, bonus)
// Per-category "any degradation", "any recovery", "any agricultural"
// ---------------------------------------------------------------
print('\n=== PART 3: Category composite rasters (3 tasks) ===');

var categories = ['deg', 'rec', 'agr'];
categories.forEach(function(cat) {
  var catTransitions = transitions.filter(function(t) { return t.category === cat; });
  
  var composite = ee.Image(0);
  catTransitions.forEach(function(t) {
    intervals.forEach(function(i) {
      composite = composite.or(
        createTransitionMap(i.startYear, i.endYear, t.from, t.to)
      );
    });
  });
  
  var catName = {deg: 'degradation', rec: 'recovery', agr: 'agricultural'}[cat];
  Export.image.toDrive({
    image: composite.rename('category_composite').toByte().clip(africaGeom),
    description: 'category_' + catName + '_cumulative',
    folder: 'PhD_Paper3_Data',
    fileNamePrefix: 'category_' + catName + '_cumulative',
    region: exportRegion,
    scale: 300,
    maxPixels: 1e13,
    crs: 'EPSG:4326'
  });
  print('  category_' + catName + '_cumulative');
});


// ---------------------------------------------------------------
// PART 4: Quick visual check
// ---------------------------------------------------------------
Map.centerObject(africaGeom, 3);

// Degradation composite
var degComposite = ee.Image(0);
transitions.filter(function(t){return t.category==='deg';}).forEach(function(t){
  intervals.forEach(function(i){
    degComposite = degComposite.or(
      createTransitionMap(i.startYear, i.endYear, t.from, t.to)
    );
  });
});
Map.addLayer(
  degComposite.selfMask().clip(africaGeom),
  {palette: ['red'], opacity: 0.7},
  'ALL Degradation transitions (cumulative 1985-2022)'
);

// Recovery composite
var recComposite = ee.Image(0);
transitions.filter(function(t){return t.category==='rec';}).forEach(function(t){
  intervals.forEach(function(i){
    recComposite = recComposite.or(
      createTransitionMap(i.startYear, i.endYear, t.from, t.to)
    );
  });
});
Map.addLayer(
  recComposite.selfMask().clip(africaGeom),
  {palette: ['green'], opacity: 0.7},
  'ALL Recovery transitions (cumulative 1985-2022)'
);

// Agricultural composite
var agrComposite = ee.Image(0);
transitions.filter(function(t){return t.category==='agr';}).forEach(function(t){
  intervals.forEach(function(i){
    agrComposite = agrComposite.or(
      createTransitionMap(i.startYear, i.endYear, t.from, t.to)
    );
  });
});
Map.addLayer(
  agrComposite.selfMask().clip(africaGeom),
  {palette: ['orange'], opacity: 0.7},
  'ALL Agricultural dynamics (cumulative 1985-2022)'
);

// ---------------------------------------------------------------
// SUMMARY
// ---------------------------------------------------------------
print('\n=== MODULE B GEE EXPORTS v2 READY ===');
print('Total tasks: 84');
print('  72 per-interval (9 transitions x 8 intervals)');
print('  9 cumulative (9 transitions x 1985-2022)');
print('  3 category composites (deg/rec/agr)');
print('');
print('If the previous 6-transition tasks are still running, CANCEL them.');
print('Go to Tasks tab and click Run on each new task.');

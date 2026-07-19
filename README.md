# Interior Plan Studio

The application turns a measured 2D floor plan into a furnished, first-person
3D walkthrough.

## Preference-driven 3D walkthrough

The walkthrough is generated directly from the measured floor plan. The
default engine uses a curated local catalog of production-authored, textured
3D furniture. The professional catalog is stored in
`assets/furniture_catalog/pro`; compact kitchen and bathroom fallbacks are
stored directly in `assets/furniture_catalog`. The assets are native meshes,
not screenshots, furniture render planes, or image-to-3D reconstructions.

The professional subset comes from Poly Haven under CC0. It includes sofas,
armchairs, tables, cabinetry, bedroom and office pieces, plants, lighting,
mirrors, framed wall art, clocks, cushions, and decorative objects. A compact
CC0 Kenney subset supplies fixture categories not present in the open
professional library. See the catalog README and license files for details.

The setup dialog controls the complete room direction:

- room type and style;
- airy, curated, or layered decoration;
- color mood and personal material/color notes;
- flooring such as oak, walnut, stone, concrete, terrazzo, or tile;
- wall treatments such as paint, limewash, timber slats, moulding, or concrete.

The renderer applies those choices to floors, focal walls, ceiling coves,
recessed lighting, modeled sconces, curtains, coordinated furniture, rugs,
textured framed artwork, mirrors, plants, and room-specific decoration. Walls,
doors, and windows remain authoritative from the floor plan. The whole-room
layout engine places the wet fixture first in every bathroom, then stages the
vanity and toilet around safe circulation. Furniture remains selectable and
rotatable in the walkthrough.

Run the desktop application with `run_app.bat`.

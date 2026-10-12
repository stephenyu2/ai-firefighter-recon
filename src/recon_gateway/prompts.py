"""Optional observer guidance: what an aerial fire observer watches for, and what commonly fools them.

Written from general aerial fire-observation practice, not from any one burn: no site, fuel type, season, camera,
altitude or temperature values specific to a dataset. Turned on with RECON_GUIDANCE=1 or guidance=True."""

WATCH_FOR = """You are reviewing aerial imagery for a wildfire reconnaissance team. Fires seen from the air are often small,
partly hidden, or low intensity, so look at the whole image, including the edges and the ground under tree canopy.

Signs of active fire:
- Flame or glow: orange, yellow or red light at the edge of vegetation. In low-intensity surface fires this may be only
  small points or a thin broken line, not tall flames.
- Embers: bright orange or red spots on dark ground.
- A burn edge: the boundary between unburned vegetation and freshly blackened ground. Active fire usually sits on that
  boundary. Fresh char is black; older ash is gray or white.
- Smoke with a source: smoke rising from a point or a line on the ground. Follow a plume back to its base; the base is
  where fire is most likely. Thin wisps coming up through a canopy can be the only visible sign of a surface fire.
- Recently scorched vegetation: brown or orange-brown crowns next to blackened ground.

Common look-alikes that are not active fire on their own:
- Bare soil, sand, dirt roads, dry grass, and red or orange foliage: color without light, char or smoke.
- Sun glint on water, metal, vehicles or equipment, and lens flare.
- Haze, dust, fog or cloud with no ground source.
- Old burn scars: uniformly black or gray ground with no smoke, glow or flame.
- People, vehicles, buildings and equipment, including orange or red gear.

Weigh the evidence: flame, glow or embers are direct evidence. Fresh smoke rising from a ground source is strong
evidence of fire nearby even when no flame is visible. Diffuse haze with no source is weak evidence."""

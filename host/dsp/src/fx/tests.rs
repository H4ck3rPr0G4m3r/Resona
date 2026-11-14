use super::*;

fn tone(n: usize, f: f32, amp: f32) -> Vec<f32> {
    (0..n).map(|i| amp * (std::f32::consts::TAU * f * i as f32 / SR).sin()).collect()
}

/// Largest |second difference| — tiny for smooth audio, large at a click or step.
fn roughness(x: &[f32]) -> f32 {
    x.windows(3).map(|w| (w[2] - 2.0 * w[1] + w[0]).abs()).fold(0.0, f32::max)
}

fn run(fx: &mut dyn Effect, x: &mut [f32]) {
    for c in x.chunks_mut(256) {
        fx.process(c);
    }
}

#[test]
fn every_effect_is_finite_and_bounded() {
    let mut rng = 1u32;
    for k in CATALOG {
        let mut fx = create(k.kind).unwrap();
        let mut x: Vec<f32> = tone(SR as usize, 180.0, 0.4)
            .into_iter()
            .map(|v| {
                rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                v + 0.05 * ((rng >> 8) as f32 / (1 << 24) as f32 - 0.5)
            })
            .collect();
        run(fx.as_mut(), &mut x);
        assert!(x.iter().all(|v| v.is_finite()), "{}", k.kind);
        assert!(x.iter().fold(0.0f32, |m, v| m.max(v.abs())) < 4.0, "{} blew up", k.kind);
    }
}

#[test]
fn sweeping_any_control_never_clicks() {
    // A UI drag: the value jumps every 64 samples (~1.3 ms), across the full range in 0.5 s.
    for k in CATALOG {
        for (i, ps) in k.params.iter().enumerate() {
            if matches!(ps.scale, Scale::Choice(_)) {
                continue;
            }
            let at = |t: f32| match ps.scale {
                Scale::Log => (ps.min.ln() + (ps.max.ln() - ps.min.ln()) * t).exp(),
                _ => ps.min + (ps.max - ps.min) * t,
            };
            let steady = |v: f32| {
                let mut fx = create(k.kind).unwrap();
                fx.set(i, v);
                fx.reset();
                let mut x = tone(24_000, 220.0, 0.3);
                run(fx.as_mut(), &mut x);
                roughness(&x[12_000..])
            };
            let reference = steady(ps.min).max(steady(ps.max)).max(1e-4);

            let mut fx = create(k.kind).unwrap();
            fx.set(i, ps.min);
            fx.reset();
            let mut warm = tone(24_000, 220.0, 0.3);
            run(fx.as_mut(), &mut warm);
            let mut x = tone(48_000, 220.0, 0.3);
            let mut worst = 0.0f32;
            let steps = 24_000 / 64;
            for (s, c) in x.chunks_mut(64).enumerate() {
                fx.set(i, at((s as f32 / steps as f32).min(1.0)));
                fx.process(c);
            }
            worst = worst.max(roughness(&x));
            assert!(
                worst <= reference * 3.0,
                "{}.{}: sweep roughness {worst:.4} vs steady {reference:.4}",
                k.kind,
                ps.key
            );
        }
    }
}

#[test]
fn limiter_holds_the_true_peak_ceiling_and_is_transparent_below_it() {
    let mut lim = Limiter::new();
    lim.set(0, -1.0);
    lim.reset();
    let mut x = tone(48_000, 997.0, 2.0); // +6 dBFS
    for (i, v) in x.iter_mut().enumerate() {
        if i % 4801 == 0 {
            *v = 3.5; // sharp isolated overs
        }
    }
    run(&mut lim, &mut x);
    let ceil = from_db_f(-1.0);
    let sample_peak = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(sample_peak <= ceil * 1.0117, "sample peak {sample_peak}");
    // True peak via 8x sinc interpolation of the output.
    let tp = true_peak(&x[2000..]);
    assert!(crate::to_db(tp) <= -1.0 + 0.5, "true peak {:.2} dBFS", crate::to_db(tp));

    let mut lim = Limiter::new();
    lim.reset();
    let src = tone(9600, 440.0, 0.2);
    let mut y = src.clone();
    run(&mut lim, &mut y);
    let d = Limiter::LATENCY;
    let err = (d..y.len()).map(|i| (y[i] - src[i - d]).abs()).fold(0.0f32, f32::max);
    assert!(err < 1e-6, "not transparent below the ceiling: {err}");
}

fn from_db_f(db: f32) -> f32 {
    crate::from_db(db)
}

fn true_peak(x: &[f32]) -> f32 {
    let mut m = 0.0f32;
    for i in 8..x.len() - 8 {
        for p in 0..8 {
            let frac = p as f32 / 8.0;
            let mut v = 0.0;
            for j in -8i32..=8 {
                let t = j as f32 - frac;
                let s = if t.abs() < 1e-6 { 1.0 } else { (std::f32::consts::PI * t).sin() / (std::f32::consts::PI * t) };
                let w = 0.5 + 0.5 * (std::f32::consts::PI * t / 9.0).cos();
                v += x[(i as i32 + j) as usize] * s * w;
            }
            m = m.max(v.abs());
        }
    }
    m
}

#[test]
fn chain_builds_rejects_unknown_and_toggles_without_steps() {
    assert!(Chain::build(&ChainSpec { slots: vec![SlotSpec { id: 1, kind: "nope".into(), ..Default::default() }] }).is_err());
    let spec = ChainSpec {
        slots: vec![SlotSpec { id: 7, kind: "gain".into(), enabled: true, params: [("gain".to_string(), 12.0)].into() }],
    };
    let mut c = Chain::build(&spec).unwrap();
    let mut x = tone(9600, 200.0, 0.1);
    c.process(&mut x[..4800]);
    c.set_enabled(7, false);
    c.process(&mut x[4800..]);
    let base = roughness(&tone(9600, 200.0, 0.4));
    assert!(roughness(&x) <= base * 1.5, "enable toggle stepped");
    assert!((x[9000].abs() - tone(9600, 200.0, 0.1)[9000].abs()).abs() < 1e-4, "bypass is exact once faded");
}

#[test]
fn live_chain_swap_crossfades_and_returns_old_chain() {
    let (mut live, ctl) = live_chain();
    let a = ChainSpec { slots: vec![SlotSpec { id: 1, kind: "gain".into(), enabled: true, params: [("gain".into(), 6.0)].into() }] };
    let b = ChainSpec { slots: vec![SlotSpec { id: 2, kind: "gain".into(), enabled: true, params: [("gain".into(), -6.0)].into() }] };
    ctl.set(a).unwrap();
    let mut x = tone(48_000, 200.0, 0.2);
    live.process(&mut x[..12_000]);
    ctl.set(b).unwrap();
    live.process(&mut x[12_000..30_000]);
    assert!(ctl.trash.lock().slots() > 0, "old chain handed back for freeing off the audio thread");
    ctl.set_param(2, "gain", 0.0).unwrap(); // in-place param change (also frees the old chain)
    live.process(&mut x[30_000..]);
    assert!(roughness(&x) <= roughness(&tone(48_000, 200.0, 0.4)) * 1.5);
    assert_eq!(ctl.trash.lock().slots(), 0);
}

#[test]
fn denoise_keeps_alignment_and_reduces_noise() {
    let mut d = create("denoise").unwrap();
    d.set(0, 0.0); // amount 0 → exactly the input, delayed by the declared latency
    d.reset();
    let src = tone(4800, 300.0, 0.3);
    let mut y = src.clone();
    run(d.as_mut(), &mut y);
    let lat = d.latency();
    assert_eq!(lat, 960);
    assert!((lat..y.len()).all(|i| (y[i] - src[i - lat]).abs() < 1e-6));

    // The denoised path must line up with the dry one, or partial amounts comb-filter and
    // everything timed off `latency()` (voice gate, A/B recorder) is off by a frame.
    let mut d = create("denoise").unwrap();
    let mut ph = 0.0f32;
    let voiced: Vec<f32> = (0..48_000)
        .map(|i| {
            ph += std::f32::consts::TAU * (140.0 + 30.0 * (i as f32 / 9000.0).sin()) / SR;
            (1..12).map(|h| (ph * h as f32).sin() / h as f32).sum::<f32>() * 0.1
        })
        .collect();
    let mut y = voiced.clone();
    run(d.as_mut(), &mut y);
    let xcorr = |lag: usize| -> f32 { (24_000..47_000).map(|i| y[i] * voiced[i - lag]).sum() };
    let best = (0..1500).max_by(|&a, &b| xcorr(a).total_cmp(&xcorr(b))).unwrap();
    assert!(best.abs_diff(lat) <= 4, "denoised output lags {best} samples, latency() says {lat}");

    let mut d = create("denoise").unwrap();
    let mut rng = 7u32;
    let mut noise: Vec<f32> = (0..48_000)
        .map(|_| {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            0.05 * (rng as f32 / u32::MAX as f32 - 0.5)
        })
        .collect();
    let e_in: f32 = noise[24_000..].iter().map(|v| v * v).sum();
    run(d.as_mut(), &mut noise);
    let e_out: f32 = noise[24_000..].iter().map(|v| v * v).sum();
    assert!(e_out < e_in * 0.25, "noise only reduced to {:.2}", e_out / e_in);
}

fn denoise_chain(enabled: bool, amount: f32) -> ChainSpec {
    ChainSpec {
        slots: vec![
            SlotSpec { id: 1, kind: "gain".into(), enabled: true, params: Default::default() },
            SlotSpec { id: 2, kind: "denoise".into(), enabled, params: [("amount".to_string(), amount)].into() },
        ],
    }
}

#[test]
fn chain_noise_suppression_costs_one_frame_and_nothing_when_bypassed() {
    // In a chain (always driven in whole 10 ms frames) RNNoise needs one frame, not two.
    let mut c = Chain::build(&denoise_chain(true, 0.0)).unwrap();
    assert_eq!(c.latency(), 480);
    let src = tone(9600, 300.0, 0.3);
    let mut y = src.clone();
    for f in y.chunks_mut(480) {
        c.process(f);
    }
    assert!((480..y.len()).all(|i| (y[i] - src[i - 480]).abs() < 1e-6), "amount 0 is the input, one frame late");

    // Bypassed, it must not delay the microphone at all (it used to cost 20 ms switched off).
    let mut c = Chain::build(&denoise_chain(false, 1.0)).unwrap();
    assert_eq!(c.latency(), 0);
    let mut y = src.clone();
    c.process(&mut y);
    assert_eq!(y, src);
}

#[test]
fn chain_denoised_path_lines_up_with_its_latency() {
    let mut c = Chain::build(&denoise_chain(true, 1.0)).unwrap();
    let mut ph = 0.0f32;
    let voiced: Vec<f32> = (0..48_000)
        .map(|i| {
            ph += std::f32::consts::TAU * (140.0 + 30.0 * (i as f32 / 9000.0).sin()) / SR;
            (1..12).map(|h| (ph * h as f32).sin() / h as f32).sum::<f32>() * 0.1
        })
        .collect();
    let mut y = voiced.clone();
    c.process(&mut y);
    let xcorr = |lag: usize| -> f32 { (24_000..47_000).map(|i| y[i] * voiced[i - lag]).sum() };
    let best = (0..1500).max_by(|&a, &b| xcorr(a).total_cmp(&xcorr(b))).unwrap();
    assert!(best.abs_diff(480) <= 4, "denoised output lags {best} samples, the chain says 480");
}

#[test]
fn toggling_noise_suppression_live_changes_latency_without_a_click() {
    let (mut live, ctl) = live_chain();
    ctl.set(denoise_chain(false, 0.0)).unwrap();
    let src = tone(96_000, 200.0, 0.2);
    let mut x = src.clone();
    let frames: Vec<&mut [f32]> = x.chunks_mut(480).collect();
    let mut lat = Vec::new();
    for (i, f) in frames.into_iter().enumerate() {
        match i {
            50 => ctl.set(denoise_chain(true, 0.0)).unwrap(),
            120 => ctl.set(denoise_chain(false, 0.0)).unwrap(),
            _ => {}
        }
        live.process(f);
        lat.push(live.latency());
    }
    assert_eq!((lat[49], lat[119], lat[199]), (0, 480, 0), "latency follows the switch");
    // Amount 0 passes the tone through, so apart from the two crossfades the output is the
    // input (shifted by the latency while on), and the crossfades themselves must be smooth.
    assert!(roughness(&x) <= roughness(&tone(9600, 200.0, 0.4)) * 1.5, "switching clicked: {}", roughness(&x));
    assert!((480 * 100..480 * 110).all(|i| (x[i] - src[i - 480]).abs() < 1e-5), "on: one frame late");
    assert!((480 * 190..480 * 199).all(|i| (x[i] - src[i]).abs() < 1e-5), "off again: no delay");
    // No dip to silence while the new chain warms up: the level never collapses mid-tone.
    let quietest = x[4800..].chunks(240).map(|c| c.iter().fold(0.0f32, |m, v| m.max(v.abs()))).fold(1.0f32, f32::min);
    assert!(quietest > 0.1, "output dipped to {quietest} during a switch");
}

#[test]
fn eq_band_hits_its_gain() {
    let mut eq = create("eq").unwrap();
    eq.set(6, 6.0); // g3: mid band (default 1.5 kHz)
    eq.reset();
    let mut x = tone(48_000, 1500.0, 0.1);
    run(eq.as_mut(), &mut x);
    let peak = x[24_000..].iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!((crate::to_db(peak / 0.1) - 6.0).abs() < 0.1, "{:.2} dB", crate::to_db(peak / 0.1));
}

/// `RESONA_DUMP_CATALOG=path cargo test -p resona-dsp dump_catalog` writes the catalog JSON
/// (used by the UI's browser preview mock).
#[test]
fn dump_catalog() {
    let json = serde_json::to_string_pretty(CATALOG).unwrap();
    assert!(json.contains("\"choice\""));
    if let Ok(p) = std::env::var("RESONA_DUMP_CATALOG") {
        std::fs::write(p, json).unwrap();
    }
}

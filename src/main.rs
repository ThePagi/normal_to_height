use bresenham::Bresenham;
use clap::Parser;

use image::imageops::FilterType::{CatmullRom, Gaussian, Triangle};
use image::{imageops, DynamicImage, GenericImageView, Rgb};
use image::{io::Reader as ImageReader, ImageBuffer, Luma};

use std::error::Error;
use std::f32::consts::PI;

use std::ops::Mul;
use std::path::Path;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    paths: Vec<String>,
    /// Should calculate height/displacement/parallax map? Will be save with _p suffix.
    #[arg(short = 'h', long = "height")]
    calculate_height: bool,
    /// Should calculate ambient occlusion map? Will be save with _ap suffix.
    #[arg(short = 'a', long = "ao")]
    calculate_ao: bool,
    /// Computes downscaled passes to better approximate global differences in height.
    #[arg(short = 'g', long = "global")]
    global_height: bool,

    /// The number of passes each pixel gets while computing noisy height.
    #[arg(short = 'i', long = "iters", default_value_t = 4)]
    noisy_iterations: i32,
    /// The range of the ray used in noisy height.
    #[arg(short = 'r', long = "range", default_value_t = 21.0)]
    noisy_range: f32,
    /// The number of passes when computing height over the axes. Removes tiling issues.
    #[arg(short = 'o', long = "overlap", default_value_t = 5)]
    overlapped_iterations: u32,
    /// Saves height to a 16bit PNG instead of 8bit.
    #[arg(long = "precise")]
    precise: bool,

    /// Exponential decay when computing height over the axes. Higher = faster decay.
    #[arg(long = "decay", default_value_t = 128.0)]
    decay: f32,
    // Exponential decay when computing AO. Higher = faster decay.
    #[arg(long = "ao_decay", default_value_t = 256.0)]
    ao_decay: f32,
    /// The amount (sigma value) of gaussian blur applied to height.
    #[arg(short = 'b', long = "blur", default_value_t = 1.0)]
    blur: f32,

    #[arg(long = "dx")]
    directx: bool,
    #[arg(short = 'd', long = "downscale", default_value_t = 1)]
    downscale: u32,
}

lazy_static::lazy_static! {
    #[derive(Debug)]
    static ref ARGS: Args = Args::parse_from(wild::args());
}

fn normalize(img: &mut ImageBuffer<Rgb<f32>, Vec<f32>>) {
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    for p in img.as_flat_samples().samples.iter() {
        if *p < min {
            min = *p;
        }
        if *p > max {
            max = *p;
        }
    }
    if (max - min).abs() > f32::EPSILON {
        img.as_flat_samples_mut()
            .samples
            .iter_mut()
            .for_each(|v| *v = (*v - min) / (max - min));
        //println!("Normalizing {min}, {max}");
    } else {
        //println!("min == max");
    }
}

fn normalize_around_zero(img: &mut ImageBuffer<Rgb<f32>, Vec<f32>>) {
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    for p in img.as_flat_samples().samples.iter() {
        if *p < min {
            min = *p;
        }
        if *p > max {
            max = *p;
        }
    }
    if (max - min).abs() > f32::EPSILON {
        img.as_flat_samples_mut()
            .samples
            .iter_mut()
            .for_each(|v| *v = 2.0 * (*v - min) / (max - min) - 1.0);
        //println!("Normalizing {min}, {max}");
    } else {
        //println!("min == max");
    }
}

fn normalize_luma(img: &mut ImageBuffer<Luma<f32>, Vec<f32>>) {
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    for p in img.as_flat_samples().samples.iter() {
        if *p < min {
            min = *p;
        }
        if *p > max {
            max = *p;
        }
    }
    if (max - min).abs() > f32::EPSILON {
        img.as_flat_samples_mut()
            .samples
            .iter_mut()
            .for_each(|v| *v = (*v - min) / (max - min));
        //println!("Normalizing {min}, {max}");
    } else {
        //println!("min == max");
    }
}

fn calc_noisy(
    img: &ImageBuffer<Rgb<f32>, Vec<f32>>,
    noisy: &mut ImageBuffer<Rgb<f32>, Vec<f32>>,
    range: f32,
    iters: i32,
) {
    let mut rng = fastrand::Rng::new();
    for yy in 0..img.height() {
        for xx in 0..img.width() {
            for _i in 0..iters {
                let dir = 2f32 * PI * rng.f32();
                let mut prevx = 0f32;
                let mut prevy = 0f32;
                let cos = dir.cos();
                let sin = dir.sin();
                let vx = (range * cos).round();
                let vy = (range * sin).round();

                for (x, y) in Bresenham::new(
                    (xx as isize + vx as isize, yy as isize + vy as isize),
                    (xx as isize, yy as isize),
                ) {
                    //mult += decay/r;
                    let x = wrapped(x as u32, img.width());
                    let y = wrapped(y as u32, img.height());
                    let hx = prevx + cos * (img.get_pixel(x, y).0[0] as f32 - 0.5);
                    let hy = prevy + sin * (img.get_pixel(x, y).0[1] as f32 - 0.5);
                    prevx = hx;
                    prevy = hy;
                    noisy.get_pixel_mut(x, y).0[0] += hx;
                    noisy.get_pixel_mut(x, y).0[1] -= hy;
                }
                noisy.get_pixel_mut(xx, yy).0[0] += prevx;
                noisy.get_pixel_mut(xx, yy).0[1] -= prevy;
            }
        }
    }
    normalize(noisy);
    //imageops::filter3x3(noisy, &[0.,0.2,0.,0.2,0.2,0.2,0.,0.2,0.])
    //noisy.clone()
}

fn calc_ddm(
    img: &ImageBuffer<Rgb<f32>, Vec<f32>>,
    decay: f32,
    offsetx: u32,
    offsety: u32,
) -> ImageBuffer<Luma<f32>, Vec<f32>> {
    let mut hor = ImageBuffer::<Rgb<f32>, Vec<f32>>::new(img.width(), img.height());
    let mut horn = ImageBuffer::<Rgb<f32>, Vec<f32>>::new(img.width(), img.height());
    let mut ver = ImageBuffer::<Rgb<f32>, Vec<f32>>::new(img.width(), img.height());
    let mut vern = ImageBuffer::<Rgb<f32>, Vec<f32>>::new(img.width(), img.height());

    //(1.0 - (16.0 / img.width() as f32));//(-16.0 / (img.width() as f32)).exp();
    //println!("{decay}");
    let mult = 2.0;
    for y in (1 + offsety)..(img.height() + offsety) {
        let y = wrapped(y, img.height());
        for x in (1 + offsetx)..(img.width() + offsetx) {
            let x = wrapped(x, img.width());
            let xl = wrapped(x - 1, img.width());
            let yl = wrapped(y - 1, img.height());
            hor.get_pixel_mut(x, y).0[0] += (hor.get_pixel(xl, y).0[0]) * decay
                + ((0.5 - img.get_pixel(xl, y).0[0] as f32) * mult);
            ver.get_pixel_mut(x, y).0[0] += (ver.get_pixel(x, yl).0[0]) * decay
                + ((img.get_pixel(x, yl).0[1] as f32 - 0.5) * mult);
            hor.get_pixel_mut(x, y).0[1] += (hor.get_pixel(xl, yl).0[1]) * decay
                + ((0.5 - img.get_pixel(xl, yl).0[0] as f32) * mult);
            ver.get_pixel_mut(x, y).0[1] += (ver.get_pixel(xl, yl).0[1]) * decay
                + ((img.get_pixel(xl, yl).0[1] as f32 - 0.5) * mult);
        }
    }

    for y in (1 + offsety)..(img.height() + offsety) {
        let y = img.height() - y - 1;
        let y = wrapped(y, img.height());
        for x in (1 + offsetx)..(img.width() + offsetx) {
            let x = img.width() - x - 1;
            let x = wrapped(x, img.width());
            let xl = wrapped(x + 1, img.width());
            let yl = wrapped(y + 1, img.height());
            horn.get_pixel_mut(x, y).0[0] += (horn.get_pixel(xl, y).0[0]) * decay
                + ((img.get_pixel(xl, y).0[0] as f32 - 0.5) * mult);
            vern.get_pixel_mut(x, y).0[0] += (vern.get_pixel(x, yl).0[0]) * decay
                + ((0.5 - img.get_pixel(x, yl).0[1] as f32) * mult);
            horn.get_pixel_mut(x, y).0[1] += (horn.get_pixel(xl, yl).0[1]) * decay
                + ((img.get_pixel(xl, yl).0[0] as f32 - 0.5) * mult);
            vern.get_pixel_mut(x, y).0[1] += (vern.get_pixel(xl, yl).0[1]) * decay
                + ((0.5 - img.get_pixel(xl, yl).0[1] as f32) * mult);
        }
    }

    //normalize_around_zero(&mut hor);
    //let mut hor = imageops::filter3x3(&hor, &[0., 0.4, 0., 0., 0.8, 0., 0., 0.4, 0.]);
    //let mut hor = imageops::blur(&hor, 1.0);
    //normalize_around_zero(&mut ver);
    //let mut ver = imageops::filter3x3(&ver, &[0., 0., 0., 0.4, 0.8, 0.4, 0., 0., 0.]);
    //let  ver = imageops::blur(&ver, 1.0);
    //normalize_around_zero(&mut horn);
    //let horn = imageops::filter3x3(&horn, &[0., 0.4, 0., 0., 0.8, 0., 0., 0.4, 0.]);
    //let mut horn = imageops::blur(&horn, 1.0);
    //normalize_around_zero(&mut vern);
    //let vern = imageops::filter3x3(&vern, &[0., 0., 0., 0.4, 0.8, 0.4, 0., 0., 0.]);
    //let  vern = imageops::blur(&ver, 1.0);

    //for y in 0..img.height() {
    //    for x in 0..img.width() {
    //        hor.get_pixel_mut(x, y).0[0] *= 1.0 - x as f32 / img.width() as f32;
    //        hor.get_pixel_mut(x, y).0[1] *= 1.0 - x as f32 / img.width() as f32;
    //        horn.get_pixel_mut(x, y).0[0] *= x as f32 / img.width() as f32;
    //        horn.get_pixel_mut(x, y).0[1] *= x as f32 / img.width() as f32;
    //        ver.get_pixel_mut(x, y).0[0] *= 1.0 - y as f32 / img.height() as f32;
    //        ver.get_pixel_mut(x, y).0[1] *= 1.0 - y as f32 / img.height() as f32;
    //        vern.get_pixel_mut(x, y).0[0] *= y as f32 / img.height() as f32;
    //        vern.get_pixel_mut(x, y).0[1] *= y as f32 / img.height() as f32;
    //    }
    //}

    let mut res = ImageBuffer::<Luma<f32>, Vec<f32>>::new(img.width(), img.height());
    for y in 0..res.height() {
        for x in 0..res.width() {
            res.get_pixel_mut(x, y).0[0] += hor.get_pixel(x, y).0.iter().sum::<f32>();
            res.get_pixel_mut(x, y).0[0] += horn.get_pixel(x, y).0.iter().sum::<f32>();
            res.get_pixel_mut(x, y).0[0] += ver.get_pixel(x, y).0.iter().sum::<f32>();
            res.get_pixel_mut(x, y).0[0] += vern.get_pixel(x, y).0.iter().sum::<f32>();
            res.get_pixel_mut(x, y).0[0] /= 8.0;
        }
    }

    for y in (offsety)..(img.height() + offsety) {
        //let y = wrapped(y, img.height());
        for x in (offsetx)..(img.width() + offsetx) {
            //let x = wrapped(x, img.width());
            let distx = 1.0
                - ((img.width() / 2 + offsetx) as f32 - x as f32).abs() * 2. / img.width() as f32;
            let disty = 1.0
                - ((img.height() / 2 + offsety) as f32 - y as f32).abs() * 2. / img.height() as f32;
            res.get_pixel_mut(wrapped(x, img.width()), wrapped(y, img.height()))
                .0[0] *= distx.min(disty).powf(0.4);
        }
    }

    //normalize_luma(&mut res);
    return res;
}

fn calc_ddm_overlapped(
    img: &ImageBuffer<Rgb<f32>, Vec<f32>>,
    decay: f32,
    count: u32,
) -> ImageBuffer<Luma<f32>, Vec<f32>> {
    let mut res = ImageBuffer::<Luma<f32>, Vec<f32>>::new(img.width(), img.height());
    for i in 0..count {
        let ddm = calc_ddm(
            img,
            decay,
            i * (7 + img.width() / count),
            i * (7 + img.height() / count),
        );
        for (j, p) in res.as_flat_samples_mut().samples.iter_mut().enumerate() {
            *p += ddm.as_flat_samples().as_slice()[j];
        }
    }
    //normalize_luma(&mut res);
    res
}

fn wrapped(ind: u32, lim: u32) -> u32 {
    ind % lim
}

fn process_ao(
    img: &ImageBuffer<Rgb<f32>, Vec<f32>>,
    filename: String,
) -> Result<(), Box<dyn Error>> {
    let mut res: ImageBuffer<Luma<f32>, Vec<f32>> = calc_ddm_overlapped(
        img,
        1.0 - (ARGS.ao_decay / img.width() as f32),
        ARGS.overlapped_iterations,
    );
    normalize_luma(&mut res);
    let vec: Vec<u8> = res
        .as_flat_samples()
        .samples
        .iter()
        .map(|v| ((*v).powf(1.1).min(0.5).mul(2.0) * u8::MAX as f32) as u8)
        .collect();
    println!("Saving: {filename}_ao.png");
    ImageBuffer::<Luma<u8>, Vec<u8>>::from_vec(img.width(), img.height(), vec)
        .unwrap()
        .save(filename + "_ao.png")?;
    Ok(())
}

fn process_height(
    origimg: &ImageBuffer<Rgb<f32>, Vec<f32>>,
    filename: String,
) -> Result<(), Box<dyn Error>> {
    let mut hmap = ImageBuffer::<Luma<f32>, Vec<f32>>::new(origimg.width(), origimg.height());
    if ARGS.global_height {
        for i in 0..(origimg.width().min(origimg.height()).ilog2()) {
            let ds = 2u32.pow(i + 1);
            let tiny = imageops::resize(
                origimg,
                origimg.width() / ds,
                origimg.height() / ds,
                CatmullRom,
            );
            let glob_blurry = imageops::resize(
                &calc_ddm_overlapped(&tiny, 1.0 - (ARGS.decay / tiny.width() as f32), ARGS.overlapped_iterations),
                origimg.width(),
                origimg.height(),
                CatmullRom,
            );
            for yy in 0..origimg.height() {
                for xx in 0..origimg.width() {
                    hmap.get_pixel_mut(xx, yy).0[0] +=
                        glob_blurry.get_pixel(xx, yy).0.iter().sum::<f32>() * ds as f32/(i as f32 +1.);
                }
            }
        }
    }

    let glob_detail = calc_ddm_overlapped(
        &origimg,
        1.0 - (ARGS.decay / origimg.width() as f32),
        ARGS.overlapped_iterations,
    );
    for yy in 0..origimg.height() {
        for xx in 0..origimg.width() {
            hmap.get_pixel_mut(xx, yy).0[0] += glob_detail.get_pixel(xx, yy).0.iter().sum::<f32>();
        }
    }
    normalize_luma(&mut hmap);
    //let mut hmap = imageops::resize(&imageops::resize(&hmap, hmap.width()/2, hmap.height()/2, Triangle), hmap.width(), hmap.height(), Triangle);

    let mut noisy_detail =
        ImageBuffer::<Rgb<f32>, Vec<f32>>::new(origimg.width(), origimg.height());
    calc_noisy(
        &origimg,
        &mut noisy_detail,
        ARGS.noisy_range * origimg.width() as f32 / 4096.0,
        ARGS.noisy_iterations,
    );
    //let noisy_detail = imageops::blur(&noisy_detail, 5.);
    let m1 = ARGS.overlapped_iterations as f32 * ARGS.decay.sqrt();
    let m2 = ARGS.noisy_iterations as f32 * ARGS.noisy_range.sqrt();
    for yy in 0..origimg.height() {
        for xx in 0..origimg.width() {
            let o = hmap.get_pixel(xx, yy).0[0];
            let n = noisy_detail.get_pixel(xx, yy).0.iter().sum::<f32>();
            hmap.get_pixel_mut(xx, yy).0[0] = o * m1 + n * m2;
        }
    }
    normalize_luma(&mut hmap);

    let mut tiledhmap =
        ImageBuffer::<Luma<f32>, Vec<f32>>::new(hmap.width() * 3, hmap.height() * 3);
    image::imageops::tile(&mut tiledhmap, &hmap);
    let mut tiledhmap = imageops::blur(&tiledhmap, ARGS.blur);
    let hmap = imageops::crop(
        &mut tiledhmap,
        hmap.width(),
        hmap.height(),
        hmap.width(),
        hmap.height(),
    )
    .to_image();

    let hmap = if ARGS.precise {
        let hvec: Vec<u16> = hmap
            .as_flat_samples()
            .samples
            .iter()
            .map(|v| (*v * u16::MAX as f32) as u16)
            .collect();
        DynamicImage::ImageLuma16(
            ImageBuffer::<Luma<u16>, Vec<u16>>::from_vec(origimg.width(), origimg.height(), hvec)
                .unwrap(),
        )
    } else {
        let hvec: Vec<u8> = hmap
            .as_flat_samples()
            .samples
            .iter()
            .map(|v| (*v * u8::MAX as f32) as u8)
            .collect();
        DynamicImage::ImageLuma8(
            ImageBuffer::<Luma<u8>, Vec<u8>>::from_vec(origimg.width(), origimg.height(), hvec)
                .unwrap(),
        )
    };
    println!("Saving: {filename}_p.png");
    hmap.save(filename.clone() + "_p.png")?;
    Ok(())
}

fn run() -> Result<(), Box<dyn Error>> {
    println!("{:#?}", *ARGS);
    for input in ARGS.paths.iter() {
        if input.ends_with("_p.png") || input.ends_with("_ao.png") {
            println!("Skipping: {input}");
            continue;
        }
        println!("Processing: {input}");
        let filename = input
            .strip_suffix(
                Path::new(&input)
                    .extension()
                    .unwrap_or_default()
                    .to_str()
                    .unwrap(),
            )
            .unwrap_or_default()
            .strip_suffix(".")
            .unwrap_or_default()
            .replace("_n", "");
        if !std::path::Path::new(input).exists() {
            println!("File {input} does not exist!");
            continue;
        }
        let img = ImageReader::open(input)?.decode()?;
        let img = img.resize(
            img.width() / ARGS.downscale,
            img.height() / ARGS.downscale,
            imageops::FilterType::Gaussian,
        );
        let mut img = img.to_rgb32f();
        if ARGS.directx {
            for p in img.pixels_mut() {
                p.0[1] = 1.0 - p.0[1];
            }
        }
        if ARGS.calculate_height {
            process_height(&img, filename.clone())?;
        }
        if ARGS.calculate_ao {
            process_ao(&img, filename)?;
        }
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        println!("Error: {}", e.to_string());
    }
}

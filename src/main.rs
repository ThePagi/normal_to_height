use bresenham::Bresenham;
use clap::Parser;

use image::imageops::FilterType::{CatmullRom, Triangle};
use image::{imageops, DynamicImage, Rgb};
use image::{io::Reader as ImageReader, ImageBuffer, Luma};



use std::error::Error;
use std::f32::consts::PI;

use std::ops::Mul;
use std::path::Path;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    paths: Vec<String>,

    #[arg(short = 'h', long = "height")]
    calculate_height: bool,
    #[arg(short = 'a', long = "ao")]
    calculate_ao: bool,
    #[arg(short = 'g', long = "global")]
    global_height: bool,

    #[arg(short = 'i', long = "iters", default_value_t = 4)]
    noisy_iterations: i32,
    #[arg(short = 'r', long = "range", default_value_t = 21.0)]
    noisy_range: f32,
    #[arg(short = 'p', long = "precise")]
    precise: bool,

    // exponential decay, decays after ~1/128 of the image
    #[arg(short = 'd', long = "decay", default_value_t = 128.0)]
    decay: f32,

    // the amount of gaussian blur (sigma)
    #[arg(short = 'b', long = "blur", default_value_t = 1.0)]
    blur: f32,
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
    img.as_flat_samples_mut()
        .samples
        .iter_mut()
        .for_each(|v| *v = (*v - min) / (max - min));
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
    img.as_flat_samples_mut()
        .samples
        .iter_mut()
        .for_each(|v| *v = (*v - min) / (max - min));
    println!("{min}, {max}");
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
) -> ImageBuffer<Rgb<f32>, Vec<f32>> {
    let mut hor = ImageBuffer::<Rgb<f32>, Vec<f32>>::new(img.width(), img.height());
    let mut horn = ImageBuffer::<Rgb<f32>, Vec<f32>>::new(img.width(), img.height());
    let mut ver = ImageBuffer::<Rgb<f32>, Vec<f32>>::new(img.width(), img.height());
    let mut vern = ImageBuffer::<Rgb<f32>, Vec<f32>>::new(img.width(), img.height());

    //(1.0 - (16.0 / img.width() as f32));//(-16.0 / (img.width() as f32)).exp();
    println!("{decay}");
    let mult = 2.0;
    for y in 1..img.height() {
        for x in 1..img.width() {
            hor.get_pixel_mut(x, y).0[0] += (hor.get_pixel(x - 1, y).0[0]) * decay
                + ((0.5 - img.get_pixel(x - 1, y).0[0] as f32) * mult);
            ver.get_pixel_mut(x, y).0[0] += (ver.get_pixel(x, y - 1).0[0]) * decay
                + ((img.get_pixel(x, y - 1).0[1] as f32 - 0.5) * mult);
            hor.get_pixel_mut(x, y).0[1] += (hor.get_pixel(x - 1, y - 1).0[1]) * decay
                + ((0.5 - img.get_pixel(x - 1, y - 1).0[0] as f32) * mult);
            ver.get_pixel_mut(x, y).0[1] += (ver.get_pixel(x - 1, y - 1).0[1]) * decay
                + ((img.get_pixel(x - 1, y - 1).0[1] as f32 - 0.5) * mult);
        }
    }

    for y in 1..img.height() {
        let y = img.height() - y - 1;
        for x in 1..img.width() {
            let x = img.width() - x - 1;
            horn.get_pixel_mut(x, y).0[0] += (horn.get_pixel(x + 1, y).0[0]) * decay
                + ((img.get_pixel(x + 1, y).0[0] as f32 - 0.5) * mult);
            vern.get_pixel_mut(x, y).0[0] += (vern.get_pixel(x, y + 1).0[0]) * decay
                + ((0.5 - img.get_pixel(x, y + 1).0[1] as f32) * mult);
            horn.get_pixel_mut(x, y).0[1] += (horn.get_pixel(x + 1, y + 1).0[1]) * decay
                + ((img.get_pixel(x + 1, y + 1).0[0] as f32 - 0.5) * mult);
            vern.get_pixel_mut(x, y).0[1] += (vern.get_pixel(x + 1, y + 1).0[1]) * decay
                + ((0.5 - img.get_pixel(x + 1, y + 1).0[1] as f32) * mult);
        }
    }

    normalize(&mut hor);
    //let mut hor = imageops::filter3x3(&hor, &[0., 0.4, 0., 0., 0.8, 0., 0., 0.4, 0.]);
    //let mut hor = imageops::blur(&hor, 1.0);
    normalize(&mut ver);
    //let mut ver = imageops::filter3x3(&ver, &[0., 0., 0., 0.4, 0.8, 0.4, 0., 0., 0.]);
    //let  ver = imageops::blur(&ver, 1.0);
    normalize(&mut horn);
    //let horn = imageops::filter3x3(&horn, &[0., 0.4, 0., 0., 0.8, 0., 0., 0.4, 0.]);
    //let mut horn = imageops::blur(&horn, 1.0);
    normalize(&mut vern);
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
    for (i, p) in hor.as_flat_samples_mut().samples.iter_mut().enumerate() {
        *p += horn.as_flat_samples().as_slice()[i];
        *p += ver.as_flat_samples().as_slice()[i];
        *p += vern.as_flat_samples().as_slice()[i];
    }

    for y in 1..hor.height() {
        for x in 1..hor.width() {
            let lx = (hor.width() as i32 / 2 - x as i32).max(0);
            let ly = (hor.height() as i32 / 2 - y as i32).max(0);
            let mut lerpx = (lx as f32 *2.0 / hor.width() as f32).powf(2.0);
            let mut lerpy = (ly as f32 *2.0 / hor.height() as f32).powf(2.0);
            lerpx /= (lerpx + lerpy).max(1.0);
            lerpy /= (lerpx + lerpy).max(1.0);
            hor.get_pixel_mut(x, y).0[0] = hor.get_pixel_mut(x, y).0[0] * (1.0 - lerpx - lerpy)
                + hor.get_pixel_mut(img.width() - x - 1, y).0[0] * lerpx
                + hor.get_pixel_mut(x, img.height() - y - 1).0[0] * lerpy;
        }
    }

    //for y in 0..img.height(){
    //    let p0 = (hor.get_pixel(0, y).0[0] + hor.get_pixel(img.width()-1, y).0[0])*0.5;
    //    let p1 = (hor.get_pixel(0, y).0[1] + hor.get_pixel(img.width()-1, y).0[1])*0.5;
    //    hor.get_pixel_mut(0, y).0[0] = p0;
    //    hor.get_pixel_mut(img.width()-1, y).0[0] = p0;
    //    hor.get_pixel_mut(0, y).0[1] = p1;
    //    hor.get_pixel_mut(img.width()-1, y).0[1] = p1;
    //}
    //for x in 0..img.width(){
    //    let p0 = (hor.get_pixel(x, 0).0[0] + hor.get_pixel(x, img.height()-1).0[0])*0.5;
    //    let p1 = (hor.get_pixel(x, 0).0[1] + hor.get_pixel(x, img.height()-1).0[1])*0.5;
    //    hor.get_pixel_mut(x, 0).0[0] = p0;
    //    hor.get_pixel_mut(x, img.height()-1).0[0] = p0;
    //    hor.get_pixel_mut(x, 0).0[1] = p1;
    //    hor.get_pixel_mut(x, img.height()-1).0[1] = p1;
    //}
    normalize(&mut hor);
    return hor;
}

fn wrapped(ind: u32, lim: u32) -> u32 {
    ind % lim
}

fn process_ao(
    img: &ImageBuffer<Rgb<f32>, Vec<f32>>, filename: String
)  -> Result<(), Box<dyn Error>>{
    let res = calc_ddm(img, 1.0 - (256.0 / img.width() as f32));
    let vec: Vec<f32> = res.pixels()
            .map(|v| v.0.iter().sum::<f32>())
            .collect();
    let mut res = ImageBuffer::<Luma<f32>, Vec<f32>>::from_vec(img.width(), img.height(), vec).unwrap();
    normalize_luma(&mut res);
    let vec: Vec<u8> = res.as_flat_samples()
    .samples.iter()
        .map(|v| ((*v).mul(2.0).min(1.0) * u8::MAX as f32) as u8)
        .collect();
    println!("Saving: {filename}_ao.png");
    ImageBuffer::<Luma<u8>, Vec<u8>>::from_vec(img.width(), img.height(), vec).unwrap().save(filename + "_ao.png")?;
    Ok(())
}

fn process_height(origimg: &ImageBuffer<Rgb<f32>, Vec<f32>>, filename: String) -> Result<(), Box<dyn Error>> {

    let glob_detail = calc_ddm(
        &origimg,
        1.0 - (ARGS.decay / origimg.width() as f32),
    );

    let mut hmap = ImageBuffer::<Luma<f32>, Vec<f32>>::new(origimg.width(), origimg.height());
    for yy in 0..origimg.height() {
        for xx in 0..origimg.width() {
            hmap.get_pixel_mut(xx, yy).0[0] = glob_detail.get_pixel(xx, yy).0.iter().sum::<f32>();
        }
    }
    if ARGS.global_height {
        let tiny = imageops::resize(
            origimg,
            origimg.width() / 128,
            origimg.height() / 128,
            Triangle,
        );
        let glob_blurry = imageops::resize(
            &calc_ddm(&tiny, 0.95),
            origimg.width(),
            origimg.height(),
            CatmullRom,
        );
        for yy in 0..origimg.height() {
            for xx in 0..origimg.width() {
                hmap.get_pixel_mut(xx, yy).0[0] +=
                    0.25 * glob_blurry.get_pixel(xx, yy).0.iter().sum::<f32>();
            }
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
    for yy in 0..origimg.height() {
        for xx in 0..origimg.width() {
            let o = hmap.get_pixel(xx, yy).0[0];
            let n = noisy_detail.get_pixel(xx, yy).0.iter().sum::<f32>();
            hmap.get_pixel_mut(xx, yy).0[0] = o + n;
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
        let filename = input.strip_suffix(Path::new(&input).extension().unwrap().to_str().unwrap())
            .unwrap()
            .strip_suffix(".").unwrap()
            .replace("_n", "");
        let img = ImageReader::open(input)?.decode()?.to_rgb32f();
        if ARGS.calculate_height{
        process_height(&img, filename.clone())?;
        }
        if ARGS.calculate_ao{
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

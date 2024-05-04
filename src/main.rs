use bresenham::Bresenham;
use either::Either;
use image::imageops::FilterType::{CatmullRom, Gaussian, Lanczos3, Nearest, Triangle};
use image::{imageops, DynamicImage, GenericImage, Rgb, Rgba, SubImage};
use image::{io::Reader as ImageReader, GenericImageView, ImageBuffer, Luma};
use itertools::Itertools;
use rand::{thread_rng, Rng};
use rand_distr::StandardNormal;
use std::error::Error;
use std::f32::consts::PI;
use std::io::Cursor;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short = 'g', long = "global")]
    global_height: bool,

    // exponential decay, decays after ~1/128 of the image
    #[arg(short = 'd', long = "decay", default_value_t = 128.0)]
    decay: f32,

    // the amount of gaussian blur (sigma)
    #[arg(short = 'b', long = "blur", default_value_t = 5.0)]
    blur: f32,
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
            for i in 0..iters {
                let dir = (2f32 * PI * rng.f32());
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
            }
        }
    }
    normalize(noisy);
    //imageops::filter3x3(noisy, &[0.,0.2,0.,0.2,0.2,0.2,0.,0.2,0.])
    //noisy.clone()
}

fn calc_ddm(
    img: &SubImage<&ImageBuffer<Rgb<f32>, Vec<f32>>>,
    decay: f32
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

    //normalize(&mut hor);
    //let mut hor = imageops::filter3x3(&hor, &[0., 0.4, 0., 0., 0.8, 0., 0., 0.4, 0.]);
    //let mut hor = imageops::blur(&hor, 1.0);
    //normalize(&mut ver);
    //let mut ver = imageops::filter3x3(&ver, &[0., 0., 0., 0.4, 0.8, 0.4, 0., 0., 0.]);
    //let  ver = imageops::blur(&ver, 1.0);
    //normalize(&mut horn);
    //let horn = imageops::filter3x3(&horn, &[0., 0.4, 0., 0., 0.8, 0., 0., 0.4, 0.]);
    //let mut horn = imageops::blur(&horn, 1.0);
    //normalize(&mut vern);
    //let vern = imageops::filter3x3(&vern, &[0., 0., 0., 0.4, 0.8, 0.4, 0., 0., 0.]);
    //let  vern = imageops::blur(&ver, 1.0);

    for (i, p) in hor.as_flat_samples_mut().samples.iter_mut().enumerate() {
        *p += horn.as_flat_samples().as_slice()[i];
        *p += ver.as_flat_samples().as_slice()[i];
        *p += vern.as_flat_samples().as_slice()[i];
    }
    normalize(&mut hor);
    return hor;
}

fn wrapped(ind: u32, lim: u32) -> u32 {
    ind % lim
}

fn main() -> Result<(), Box<dyn Error>> {
    let origimg = ImageReader::open("rocks.png")?.decode()?.to_rgb32f();

    let glob_detail = calc_ddm(
        &origimg.view(0, 0, origimg.width(), origimg.height()),
        1.0 - (128.0 / origimg.width() as f32));
    let tiny = imageops::resize(&origimg, origimg.width()/128, origimg.height()/128, Triangle);
    let glob_blurry = imageops::resize(&calc_ddm(
        &tiny.view(0, 0, tiny.width(), tiny.height()),
        0.95), origimg.width(), origimg.height(), CatmullRom);
    let mut hmap = ImageBuffer::<Luma<f32>, Vec<f32>>::new(origimg.width(), origimg.height());
    for yy in 0..origimg.height() {
        for xx in 0..origimg.width() {
            hmap.get_pixel_mut(xx, yy).0[0] = glob_detail.get_pixel(xx, yy).0.iter().sum::<f32>() + 0.25*glob_blurry.get_pixel(xx, yy).0.iter().sum::<f32>();
        }
    }
    normalize_luma(&mut hmap);
    //let mut hmap = imageops::resize(&imageops::resize(&hmap, hmap.width()/2, hmap.height()/2, Triangle), hmap.width(), hmap.height(), Triangle);

    let mut noisy_detail = ImageBuffer::<Rgb<f32>, Vec<f32>>::new(origimg.width(), origimg.height());
    calc_noisy(&origimg, &mut noisy_detail, origimg.width() as f32 / 196.0, 4);
    //let noisy_detail = imageops::blur(&noisy_detail, 5.);
    for yy in 0..origimg.height() {
        for xx in 0..origimg.width() {
            hmap.get_pixel_mut(xx, yy).0[0] *= noisy_detail.get_pixel(xx, yy).0.iter().sum::<f32>();
        }
    }
    normalize_luma(&mut hmap);

    let mut tiledhmap =
        ImageBuffer::<Luma<f32>, Vec<f32>>::new(hmap.width() * 3, hmap.height() * 3);
    image::imageops::tile(&mut tiledhmap, &hmap);
    let mut tiledhmap = imageops::blur(&tiledhmap, 1.0);
    let hmap = imageops::crop(&mut tiledhmap, hmap.width(), hmap.height(), hmap.width(), hmap.height()).to_image();
    let hvec: Vec<u16> = hmap
        .as_flat_samples()
        .samples
        .iter()
        .map(|v| (*v * (u16::MAX as f32 - 1.0)) as u16)
        .collect();
    let mut hmap = DynamicImage::ImageLuma16(
        ImageBuffer::<Luma<u16>, Vec<u16>>::from_vec(origimg.width(), origimg.height(), hvec).unwrap())/*.blur(1.0)*/;
    hmap.save("h.tif").unwrap();
    Ok(())
}

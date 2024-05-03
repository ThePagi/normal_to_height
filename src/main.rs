use bresenham::Bresenham;
use either::Either;
use image::{imageops, DynamicImage, Rgb, Rgba, SubImage};
use image::{io::Reader as ImageReader, GenericImageView, ImageBuffer, Luma};
use itertools::Itertools;
use rand::{thread_rng, Rng};
use rand_distr::StandardNormal;
use std::error::Error;
use std::f32::consts::PI;
use std::io::Cursor;

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

fn calc_noisy(img: &SubImage<&ImageBuffer<Rgb<f32>, Vec<f32>>>, noisy: &mut ImageBuffer::<Rgb<f32>, Vec<f32>>) -> ImageBuffer::<Rgb<f32>, Vec<f32>>{
    let mut rng = fastrand::Rng::new();
    let r = img.width() as f32 / 64.0;
    for yy in 0..img.height() {
        for xx in 0..img.width() {
            for i in 0..1 {
                let dir = (2f32 * PI * rng.f32());
                let mut prevx = 0f32;
                let mut prevy = 0f32;
                let cos = dir.cos();
                let sin = dir.sin();
                let vx = (r * cos).round();
                let vy = (r * sin).round();

                for (x, y) in Bresenham::new(
                    (
                        0.max((img.width() as isize - 1).min(xx as isize + vx as isize)),
                        0.max((img.height() as isize - 1).min(yy as isize + vy as isize)),
                    ),
                    (xx as isize, yy as isize),
                ) {
                    //mult += decay/r;
                    let x = x as u32;
                    let y = y as u32;
                    let hx = prevx + cos
                    * (img.get_pixel(x, y).0[0] as f32
                        - 0.5);
                    let hy = prevy + sin
                    * (img.get_pixel(x, y).0[1] as f32
                        - 0.5)
                    ;
                    prevx = hx;
                    prevy = hy;
                    noisy.get_pixel_mut(x, y).0[0] += hx;
                    noisy.get_pixel_mut(x, y).0[1] -= hy;
                }
            }
        }
    }
    normalize(noisy);
    noisy.clone()
    //imageops::blur(noisy, 1.0)
}

fn calc_ddm(img: &SubImage<&ImageBuffer<Rgb<f32>, Vec<f32>>>, border: u32) -> ImageBuffer<Rgb<f32>, Vec<f32>> {
    let mut hor = ImageBuffer::<Rgb<f32>, Vec<f32>>::new(img.width(), img.height());
    let mut horn = ImageBuffer::<Rgb<f32>, Vec<f32>>::new(img.width(), img.height());
    let mut ver = ImageBuffer::<Rgb<f32>, Vec<f32>>::new(img.width(), img.height());
    let mut vern = ImageBuffer::<Rgb<f32>, Vec<f32>>::new(img.width(), img.height());
    let mut noisy = ImageBuffer::<Rgb<f32>, Vec<f32>>::new(img.width(), img.height());

    let decay = (-16.0 / (img.width() as f32)).exp();
    println!("{decay}");
    let mult = 0.1;
    let power = 0.99999;
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
                + ((img.get_pixel(x + 1, y+1).0[0] as f32 - 0.5) * mult);
            vern.get_pixel_mut(x, y).0[1] += (vern.get_pixel(x + 1, y + 1).0[1]) * decay
                + ((0.5 - img.get_pixel(x+1, y + 1).0[1] as f32) * mult);
        }
    }

    
    normalize(&mut hor);
    let mut hor = imageops::filter3x3(&hor, &[0., 0.4, 0.2, 0., 0.8, 0., 0.2, 0.4, 0.]);
    //let mut hor = imageops::blur(&hor, 1.0);
    normalize(&mut ver);
    let mut ver = imageops::filter3x3(&ver, &[0., 0., 0.2, 0.4, 0.8, 0.4, 0.2, 0., 0.]);
    //let  ver = imageops::blur(&ver, 1.0);
    normalize(&mut horn);
    let horn = imageops::filter3x3(&horn, &[0., 0.4, 0.2, 0., 0.8, 0., 0.2, 0.4, 0.]);
    //let mut horn = imageops::blur(&horn, 1.0);
    normalize(&mut vern);
    let vern = imageops::filter3x3(&vern, &[0., 0., 0.2, 0.4, 0.8, 0.4, 0.2, 0., 0.]);
    //let  vern = imageops::blur(&ver, 1.0);

    let noisy = calc_noisy(img, &mut noisy);
    
    for (i, p) in hor.as_flat_samples_mut().samples.iter_mut().enumerate() {
        //*p = *[*p, horn.as_flat_samples().as_slice()[i], ver.as_flat_samples().as_slice()[i], vern.as_flat_samples().as_slice()[i]].iter().min_by(|x,y| (*x-0.5).abs().total_cmp(&(*y-0.5).abs())).unwrap();
        *p += horn.as_flat_samples().as_slice()[i];
        *p += ver.as_flat_samples().as_slice()[i];
        *p += vern.as_flat_samples().as_slice()[i];
        *p += noisy.as_flat_samples().as_slice()[i]*3.0;
    }
    
    return hor;
}

fn wrapped(ind: u32, lim: u32) -> u32 {
    ind % lim
}

fn main() -> Result<(), Box<dyn Error>> {
    let origimg = ImageReader::open("rocks.png")?.decode()?.to_rgb32f();
    let mut tiledimg =
        ImageBuffer::<Rgb<f32>, Vec<f32>>::new(origimg.width() * 3, origimg.height() * 3);
    image::imageops::tile(&mut tiledimg, &origimg);
    //println!("{:?}", img);
    let border = 8;
    let img = tiledimg.view(
        origimg.width() - border,
        origimg.height() - border,
        origimg.width() + border * 2,
        origimg.height() + border * 2,
    );
    let ddm = calc_ddm(&img, border);
    let mut hmap = ImageBuffer::<Luma<f32>, Vec<f32>>::new(img.width(), img.height());
    let mut rng = thread_rng();
    let g = wrapped;
    for yy in 0..img.height() {
        for xx in 0..img.width() {
            let mut acc = 0f32;
            let p = ddm.get_pixel(xx, yy).0;
            //acc = p[0] * (img.width() - xx) as f32
            //    + p[2] * (xx) as f32
            //    + p[1] * (img.height() - yy) as f32
            //    + p[3] * (yy) as f32;
            //acc /= (img.width() + img.height()) as f32;
            acc = p.iter().sum();

            hmap.get_pixel_mut(xx, yy).0[0] = acc;
        }
    }
    let mut hmap =
        imageops::crop(&mut hmap, border, border, origimg.width(), origimg.height()).to_image();
    normalize_luma(&mut hmap);
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

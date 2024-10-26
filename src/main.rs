use std::{
    collections::{HashMap, HashSet},
    hash::{DefaultHasher, Hash as _, Hasher as _},
    ops::{self, Add, Deref},
    str::FromStr,
};

use cmder::Program;
use rand::{seq::SliceRandom as _, thread_rng, Rng as _, SeedableRng as _};

use rand_xoshiro::Xoshiro256StarStar;
use smallvec::SmallVec;

fn main() {
    let mut program = Program::new();
    program
        .version(env!("CARGO_PKG_VERSION"))
        .description(env!("CARGO_PKG_DESCRIPTION"))
        .author(env!("CARGO_PKG_AUTHORS"));

    program
        .option("--width <WIDTH>", "Width of the image, default is 1920")
        .option("--height <HEIGHT>", "Height of the image, default is 1080")
        .option(
            "-s --seed",
            "Seed for the random number generator, default is random",
        )
        .option(
            "-p --points",
            "Number of points to generate, default is 1% of the image size",
        )
        .argument("output", "Output image file path")
        .action(|args| {
            let width = args
                .get_option_arg("<WIDTH>")
                .map(|w| w.parse::<i32>().unwrap())
                .unwrap_or(1920);
            let height = args
                .get_option_arg("<HEIGHT>")
                .map(|h| h.parse::<i32>().unwrap())
                .unwrap_or(1080);

            let seed = args
                .get_option_arg("seed")
                .map(|s| {
                    s.parse::<u64>()
                        .expect("seed mut be a 64bit unsigned integer")
                })
                .unwrap_or_else(|| thread_rng().gen());

            let point_count = args
                .get_option_arg("points")
                .map(|p| p.parse::<u8>().expect("points must be a 8bit unsigned integer bigger than 0 and less then size of the image"))
                .unwrap_or_else(|| {
                    let size = Dims(width, height);
                    (size.product() / 100).min(u8::MAX as i32) as u8
                });

            let output = args.get_arg("output").expect("output file path is required");

            let size = Dims(width, height);
            let mut rng = Xoshiro256StarStar::seed_from_u64(seed);

            println!("Generating image: {:?}", (size, seed, point_count));
            println!("Generating points");
            let points = randon_points(size, point_count, &mut rng);

            println!("Generating groups");
            let groups = split_groups(points, size, &mut rng);

            save_output(size, groups, seed, &output);
            println!("Image saved as output.png");
        });

    program.parse();

    // let seed = get_arg(&args, "s", Some(0u64));
    // let point_count = get_arg(&args, "p", Some(0u8));
    //
    // let seed = if seed == 0 { thread_rng().gen() } else { seed };
    //
    // let size = Dims(width, height);
    // let point_count = if point_count == 0 {
    //     (size.product() / 100).min(u8::MAX as i32) as u8
    // } else {
    //     point_count
    // };
    //
    // let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
    //
    // println!("Generating points: {:?}", (size, point_count));
    // let points = randon_points(size, point_count, &mut rng);
    //
    // println!("Generating groups");
    // let groups = split_groups(points, size, &mut rng);
    //
    // save_output(size, groups, seed);
}

fn save_output(size: Dims, groups: Array2D<u8>, base_hash: u64, filename: &str) {
    let mut img = image::ImageBuffer::new(size.0 as u32, size.1 as u32);

    img.enumerate_pixels_mut().for_each(|(x, y, pixel)| {
        let cell = Dims(x as i32, y as i32);
        let group = groups.get(cell).unwrap();

        let mut hasher = DefaultHasher::new();
        group.hash(&mut hasher);

        let hash = hasher.finish().wrapping_add(base_hash);
        let (r, g, b) = ((hash >> 16) as u8, (hash >> 8) as u8, hash as u8);

        *pixel = image::Rgb([r, g, b]);
    });

    img.save(filename).unwrap();
}

fn randon_points(size: Dims, count: u8, rng: &mut Xoshiro256StarStar) -> Vec<Dims> {
    assert!(size.all_positive());
    assert!(count as i32 <= size.product());

    let count = count as usize;
    let mut points = Vec::with_capacity(count);

    rng.gen_range(0..size.0);

    while points.len() < count {
        let point = Dims(rng.gen_range(0..size.0), rng.gen_range(0..size.1));

        if !points.contains(&point) {
            points.push(point);
        }
    }

    points
}

fn split_groups(points: Vec<Dims>, size: Dims, rng: &mut Xoshiro256StarStar) -> Array2D<u8> {
    assert!(points.len() <= u8::MAX as usize);
    assert!(!points.is_empty());
    assert!(points.clone().into_iter().collect::<HashSet<_>>().len() == points.len());

    let mut groups = Array2D::new_dims(None, size).unwrap();

    for (i, point) in points.into_iter().enumerate() {
        groups[point] = Some((i as u8, usize::MAX));
    }

    let mut cycle = 0usize;

    loop {
        if groups.all(|group| group.is_some()) {
            break;
        }

        for cell in Dims::iter_fill(Dims::ZERO, size) {
            if groups[cell].is_some() {
                continue;
            }

            let neighbors = CellWall::get_in_order()
                .into_iter()
                .map(|dir| cell + dir.to_coord())
                .filter_map(|pos| groups.get(pos).and_then(|g| g.map(|(g, _)| g)))
                .collect::<SmallVec<[_; 6]>>();

            if let Some(new_group) = neighbors.choose(rng) {
                groups[cell] = Some((*new_group, cycle));
            }
        }

        cycle = cycle.wrapping_add(1);
    }

    groups.map(|group| group.unwrap().0)
}

#[derive(Copy, Clone)]
pub enum CellWall {
    Left,
    Right,
    Top,
    Bottom,
    Up,
    Down,
}

impl CellWall {
    fn to_coord(self) -> Dims {
        match self {
            Self::Left => Dims(-1, 0),
            Self::Right => Dims(1, 0),
            Self::Top => Dims(0, -1),
            Self::Bottom => Dims(0, 1),
            Self::Up => Dims(0, 0),
            Self::Down => Dims(0, 0),
        }
    }

    fn get_in_order() -> [CellWall; 6] {
        use CellWall::*;
        [Top, Left, Right, Bottom, Up, Down]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Dims(i32, i32);

impl Add for Dims {
    type Output = Dims;

    fn add(self, other: Dims) -> Dims {
        Dims(self.0 + other.0, self.1 + other.1)
    }
}

impl Dims {
    const ZERO: Dims = Dims(0, 0);

    fn iter_fill(from: Dims, to: Dims) -> impl Iterator<Item = Dims> {
        (from.0..to.0).flat_map(move |x| (from.1..to.1).map(move |y| Dims(x, y)))
    }

    fn all_positive(self) -> bool {
        self.0 > 0 && self.1 > 0
    }

    fn all_non_negative(self) -> bool {
        self.0 >= 0 && self.1 >= 0
    }

    fn product(self) -> i32 {
        self.0 * self.1
    }
}

#[derive(Debug, Clone)]
struct Array2D<T> {
    buf: Vec<T>,
    width: usize,
    height: usize,
}

impl<T> Array2D<T> {
    fn len(&self) -> usize {
        self.buf.len()
    }

    fn dim_to_idx(&self, pos: Dims) -> Option<usize> {
        let Dims(x, y) = pos;
        let (x, y) = (x as usize, y as usize);

        if x >= self.width || y >= self.height {
            return None;
        }

        Some(y * self.width + x)
    }

    fn idx_to_dim(&self, idx: usize) -> Option<Dims> {
        if idx >= self.buf.len() {
            return None;
        }

        let x = idx % self.width;
        let y = (idx / self.width) % self.height;

        Some(Dims(x as i32, y as i32))
    }

    fn get(&self, pos: Dims) -> Option<&T> {
        self.dim_to_idx(pos).and_then(|i| self.buf.get(i))
    }
}

impl<T> Array2D<T> {
    fn all(&self, f: impl Fn(&T) -> bool) -> bool {
        self.buf.iter().all(f)
    }

    fn map<U>(self, f: impl Fn(T) -> U) -> Array2D<U> {
        Array2D {
            buf: self.buf.into_iter().map(f).collect(),
            width: self.width,
            height: self.height,
        }
    }
}

impl<T: Clone> Array2D<T> {
    fn new(item: T, width: usize, height: usize) -> Self {
        Self {
            buf: vec![item.clone(); width * height],
            width,
            height,
        }
    }

    fn new_dims(item: T, size: Dims) -> Option<Self> {
        if !size.all_non_negative() {
            return None;
        }
        Some(Self::new(item, size.0 as usize, size.1 as usize))
    }
}

impl<T> ops::Index<Dims> for Array2D<T> {
    type Output = T;

    fn index(&self, index: Dims) -> &Self::Output {
        self.dim_to_idx(index)
            .and_then(|i| self.buf.get(i))
            .expect("Index out of bounds")
    }
}

impl<T> ops::IndexMut<Dims> for Array2D<T> {
    fn index_mut(&mut self, index: Dims) -> &mut Self::Output {
        self.dim_to_idx(index)
            .and_then(|i| self.buf.get_mut(i))
            .expect("Index out of bounds")
    }
}

use hashbrown::HashMap;
use rand::{Rng, RngExt};
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Arc;

const END: usize = usize::MAX;
const MAGIC: &[u8; 5] = b"MKVC\x01";

fn invalid_data() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, "invalid markov chain file")
}

fn read_u64(reader: &mut impl Read) -> io::Result<u64> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_usize(reader: &mut impl Read) -> io::Result<usize> {
    usize::try_from(read_u64(reader)?).map_err(|_| invalid_data())
}

fn write_usize(writer: &mut impl Write, value: usize) -> io::Result<()> {
    writer.write_all(&u64::try_from(value).map_err(|_| invalid_data())?.to_le_bytes())
}

fn read_word(reader: &mut impl Read) -> io::Result<Arc<str>> {
    let word_len = read_usize(reader)?;
    let mut bytes = Vec::new();
    reader.take(word_len as u64).read_to_end(&mut bytes)?;
    if bytes.len() != word_len {
        return Err(invalid_data());
    }
    let word: Arc<str> = String::from_utf8(bytes).map_err(|_| invalid_data())?.into();
    if word.is_empty() {
        return Err(invalid_data());
    }
    Ok(word)
}

fn read_context(reader: &mut impl Read, order: usize, word_count: usize) -> io::Result<Vec<usize>> {
    let context_len = read_usize(reader)?;
    if context_len > order {
        return Err(invalid_data());
    }
    let mut context = Vec::with_capacity(context_len);
    for _ in 0..context_len {
        let id = read_usize(reader)?;
        if id >= word_count {
            return Err(invalid_data());
        }
        context.push(id);
    }
    Ok(context)
}

#[derive(Default)]
struct Transitions {
    counts: HashMap<usize, u64>,
    total: u64,
}

impl Transitions {
    fn read(reader: &mut impl Read, word_count: usize) -> io::Result<Self> {
        let edge_count = read_usize(reader)?;
        if edge_count == 0 {
            return Err(invalid_data());
        }
        let mut transitions = Self::default();
        for _ in 0..edge_count {
            let next = read_usize(reader)?;
            let count = read_u64(reader)?;
            if next != END && next >= word_count {
                return Err(invalid_data());
            }
            if count == 0 || transitions.counts.contains_key(&next) {
                return Err(invalid_data());
            }
            transitions.total = transitions.total.checked_add(count).ok_or_else(invalid_data)?;
            transitions.counts.insert(next, count);
        }
        Ok(transitions)
    }

    fn add(&mut self, next: usize) {
        *self.counts.entry(next).or_default() += 1;
        self.total += 1;
    }

    fn choose<R: Rng + ?Sized>(&self, rng: &mut R) -> usize {
        if self.counts.len() == 1 {
            return *self.counts.keys().next().unwrap();
        }
        let mut choice = rng.random_range(0..self.total);
        for (&next, &count) in &self.counts {
            if choice < count {
                return next;
            }
            choice -= count;
        }
        unreachable!()
    }
}

pub struct MarkovChain {
    order: NonZeroUsize,
    words: Vec<Arc<str>>,
    word_ids: HashMap<Arc<str>, usize>,
    transitions: HashMap<Vec<usize>, Transitions>,
}

impl MarkovChain {
    /// creates a model preceding words per transition
    pub fn new(order: NonZeroUsize) -> Self {
        Self { order, words: Vec::new(), word_ids: HashMap::new(), transitions: HashMap::new() }
    }

    /// trains on a separate seq
    pub fn train<I, S>(&mut self, words: I)
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut context = Vec::with_capacity(self.order.get());

        for word in words {
            let word = word.as_ref();
            if word.is_empty() {
                continue;
            }
            let id = match self.word_ids.get(word) {
                Some(&id) => id,
                None => {
                    let id = self.words.len();
                    let word: Arc<str> = word.into();
                    self.words.push(Arc::clone(&word));
                    self.word_ids.insert(word, id);
                    id
                }
            };
            self.transitions.entry_ref(context.as_slice()).or_default().add(id);
            if context.len() == self.order.get() {
                context.remove(0);
            }
            context.push(id);
        }

        if !context.is_empty() {
            self.transitions.entry(context).or_default().add(END);
        }
    }

    /// trains on a seq split
    pub fn train_text(&mut self, text: &str) {
        self.train(text.split_whitespace());
    }

    pub fn save(&self, mut writer: impl Write) -> io::Result<()> {
        writer.write_all(MAGIC)?;
        write_usize(&mut writer, self.order.get())?;
        write_usize(&mut writer, self.words.len())?;
        for word in &self.words {
            write_usize(&mut writer, word.len())?;
            writer.write_all(word.as_bytes())?;
        }
        write_usize(&mut writer, self.transitions.len())?;
        for (context, transitions) in &self.transitions {
            write_usize(&mut writer, context.len())?;
            for &id in context {
                write_usize(&mut writer, id)?;
            }
            write_usize(&mut writer, transitions.counts.len())?;
            for (&next, &count) in &transitions.counts {
                write_usize(&mut writer, next)?;
                writer.write_all(&count.to_le_bytes())?;
            }
        }
        Ok(())
    }

    pub fn load(mut reader: impl Read) -> io::Result<Self> {
        let mut magic = [0; 5];
        reader.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(invalid_data());
        }
        let order = NonZeroUsize::new(read_usize(&mut reader)?).ok_or_else(invalid_data)?;
        let mut chain = Self::new(order);
        let word_count = read_usize(&mut reader)?;
        for _ in 0..word_count {
            let word = read_word(&mut reader)?;
            if chain.word_ids.contains_key(word.as_ref()) {
                return Err(invalid_data());
            }
            chain.word_ids.insert(Arc::clone(&word), chain.words.len());
            chain.words.push(word);
        }
        let state_count = read_usize(&mut reader)?;
        for _ in 0..state_count {
            let context = read_context(&mut reader, order.get(), word_count)?;
            let transitions = Transitions::read(&mut reader, word_count)?;
            if chain.transitions.insert(context, transitions).is_some() {
                return Err(invalid_data());
            }
        }
        let mut trailing = [0];
        if reader.read(&mut trailing)? != 0 {
            return Err(invalid_data());
        }
        Ok(chain)
    }

    pub fn save_to_path(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let mut writer = BufWriter::new(File::create(path)?);
        self.save(&mut writer)?;
        writer.flush()
    }

    pub fn load_from_path(path: impl AsRef<Path>) -> io::Result<Self> {
        Self::load(BufReader::new(File::open(path)?))
    }

    pub fn generate<R: Rng + ?Sized>(&self, rng: &mut R, max_words: usize) -> Vec<&str> {
        let mut words = Vec::new();
        for _ in 0..max_words {
            let context = &words[words.len().saturating_sub(self.order.get())..];
            let Some(transitions) = self.transitions.get(context) else {
                break;
            };
            let next = transitions.choose(rng);
            if next == END {
                break;
            }
            words.push(next);
        }
        words.into_iter().map(|id| self.words[id].as_ref()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    fn chain(order: usize) -> MarkovChain {
        MarkovChain::new(NonZeroUsize::new(order).unwrap())
    }

    #[test]
    fn empty_model_and_zero_limit() {
        let mut rng = StdRng::seed_from_u64(1);
        let mut model = chain(2);
        assert!(model.generate(&mut rng, 10).is_empty());
        model.train_text("один два");
        assert!(model.generate(&mut rng, 0).is_empty());
    }

    #[test]
    fn reproduces_single_sequence_with_unicode() {
        let mut model = chain(2);
        model.train_text("привет, большой мир!");
        let mut rng = StdRng::seed_from_u64(1);
        assert_eq!(model.generate(&mut rng, 10), ["привет,", "большой", "мир!"]);
    }

    #[test]
    fn respects_sequence_boundaries_and_short_inputs() {
        let mut model = chain(3);
        model.train_text("один");
        model.train_text("два три");
        model.train_text("  ");
        let mut rng = StdRng::seed_from_u64(9);
        for _ in 0..100 {
            let generated = model.generate(&mut rng, 10);
            assert!(generated == ["один"] || generated == ["два", "три"]);
        }
    }

    #[test]
    fn bounds_cycles() {
        let mut model = chain(1);
        model.train_text("а а а а");
        let mut rng = StdRng::seed_from_u64(2);
        for _ in 0..100 {
            assert!(model.generate(&mut rng, 3).len() <= 3);
        }
    }

    #[test]
    fn counts_repeated_transitions() {
        let mut model = chain(1);
        model.train_text("а б");
        model.train_text("а б");
        model.train_text("а в");
        let a = model.word_ids["а"];
        let b = model.word_ids["б"];
        let v = model.word_ids["в"];
        let transitions = &model.transitions[&vec![a]];
        assert_eq!(transitions.counts[&b], 2);
        assert_eq!(transitions.counts[&v], 1);
    }

    #[test]
    fn saved_chain_can_be_trained_again() {
        let mut model = chain(2);
        model.train_text("кот спит");
        let mut bytes = Vec::new();
        model.save(&mut bytes).unwrap();
        let mut loaded = MarkovChain::load(bytes.as_slice()).unwrap();
        loaded.train_text("кот ест");
        let kot = loaded.word_ids["кот"];
        let transitions = &loaded.transitions[&vec![kot]];
        assert_eq!(transitions.counts[&loaded.word_ids["спит"]], 1);
        assert_eq!(transitions.counts[&loaded.word_ids["ест"]], 1);
        assert_eq!(loaded.order.get(), 2);
    }

    #[test]
    fn rejects_truncated_or_corrupt_file() {
        let mut model = chain(1);
        model.train_text("a b");
        let mut bytes = Vec::new();
        model.save(&mut bytes).unwrap();
        assert!(MarkovChain::load(&bytes[..bytes.len() - 1]).is_err());
        bytes[0] = 0;
        assert!(MarkovChain::load(bytes.as_slice()).is_err());
    }
}

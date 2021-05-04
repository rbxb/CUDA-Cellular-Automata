#ifndef SAVE_FRAME_H
#define SAVE_FRAME_H

class SaveFrame {
public:
	virtual bool save_frame(int i) { return true; };
};

#endif // SAVE_FRAME_H

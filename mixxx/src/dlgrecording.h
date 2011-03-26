#ifndef DLGRECORDING_H
#define DLGRECORDING_H

#include <QItemSelection>
#include "ui_dlgrecording.h"
#include "configobject.h"
#include "trackinfoobject.h"
#include "library/libraryview.h"
#include "library/trackcollection.h"
#include "library/browse/browsetablemodel.h"

class PlaylistTableModel;
class WTrackTableView;
class AnalyserQueue;
class QSqlTableModel;
class ControlObjectThreadMain;

class DlgRecording : public QWidget, public Ui::DlgRecording, public virtual LibraryView {
    Q_OBJECT
  public:
    DlgRecording(QWidget *parent, ConfigObject<ConfigValue>* pConfig, TrackCollection* pTrackCollection);
    virtual ~DlgRecording();

    virtual void setup(QDomNode node);
    virtual void onSearchStarting();
    virtual void onSearchCleared();
    virtual void onSearch(const QString& text);
    virtual void onShow();
    virtual void loadSelectedTrack();
    virtual void loadSelectedTrackToGroup(QString group);
    virtual void moveSelection(int delta);

  public slots:
    void toggleRecording(bool toggle);

  signals:
    void loadTrack(TrackPointer tio);
    void loadTrackToPlayer(TrackPointer tio, QString group);

  private:

    ConfigObject<ConfigValue>* m_pConfig;
    TrackCollection* m_pTrackCollection;
    WTrackTableView* m_pTrackTableView;
    BrowseTableModel m_browseTableModel;

    bool m_bAutoDJEnabled;
    ControlObjectThreadMain* m_pRecordingCO;

};

#endif //DLGRECORDING_H


